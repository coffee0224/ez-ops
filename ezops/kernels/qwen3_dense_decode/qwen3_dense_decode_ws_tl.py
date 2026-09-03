"""Warp-specialized standalone TileLang kernels for each qwen3_dense_decode op.

V2 (shared page pool): every op speaks the SAME shared-memory protocol so the
ops can be merged behind one pool in the megakernel-form persistent backend
(tilelang_persistent_ws) — mirroring how pdl-megakernel-reconstruction shares
a flat page pool across op modules:

  * one Pool of NP pages, each page a (64, 128) bf16 tile (16 KB):
      - gemv ops  : one page = one (BN=64 rows, BK=128 cols) weight tile
      - qkv       : one (D=128, BK=128) tile = two stacked pages (row halves)
      - attention : K tile (BS=64, D=128) = 1 page, V tile = 1 page, and the
                    Q tile (BH=64, D=128) = 1 page staged by the LOADER
  * a single global mbarrier ring over the pages, walked in tile order g:
      loader   fill g -> page p = g % NP:
          if g >= NP: wait FREE[p]  parity ((g // NP) - 1) & 1
          copy ... arrive READY[p]
      consumer tile g:
          wait READY[p] parity (g // NP) & 1
          compute ...  arrive FREE[p]
    Hardware phase bits make cross-op / cross-round reuse drift-free (the
    parity formulas were validated on odd and even NP with a stress probe).
  * loader uses plain T.copy (NOT T.async_copy): a wait_group anywhere in the
    loader makes tilelang emit a (3, 32) partial barrier that collides with
    the consumer-side (3, 128) reductions — the two ThreadSync passes assign
    named-barrier ids independently and both start at 3. That collision is
    the root cause of the V1 intermittent launch deadlock; _assert_safe_ids
    below now rejects any id used with two different counts. A T.copy loader
    warp still streams at 86% of HBM peak (386 GB/s measured on sm_120).
  * consumers never WRITE shared memory: all smem writes are loader-side, so
    every compiler-inserted partial sync stays inside one role's thread scope
    (loader 32 / consumer 128) with consistent counts.

Warp layout: threads = (NLOAD + NCONS) * 32 = 160 (1 loader + 4 consumers).

Ops (matching the persistent backend's phase semantics):
  op1 qkv     : rmsnorm(hidden) @ Wqkv.T + q/k head-norm + KV write-back
  op2 attn    : split flash-decoding GQA partials (K/V/Q staged by loader)
  op3 combine : LSE-weighted split merge (thread-parallel, no roles)
  op4 o_proj  : x2 = hidden + attn @ Wo.T
  op5 gate_up : gu = rmsnorm(x2) @ Wgu.T
  op6 down    : out = x2 + (silu(g)*u) @ Wd.T
"""

import logging
import math
import re

import tilelang
import torch
from tilelang import PassConfigKey
from tilelang import language as T
from tilelang.carver.arch.driver.cuda_driver import get_num_sms

logging.getLogger("tilelang.cache.kernel_cache").setLevel(logging.ERROR)

RMS_EPS = 1e-6

NLOAD = 1   # loader warps
NCONS = 4   # consumer warps
THREADS = (NLOAD + NCONS) * 32
NCONS_THREADS = NCONS * 32

NPAGES = 5  # pool depth (pages of 16 KB); probe: NP in 3..5 all bandwidth-bound

# Page geometry: (rows, cols) of one pool page.
PR, PC = 64, 128


def _assert_safe_barrier_ids(kernel) -> None:
    """Post-compile safety check on the generated named-barrier usage.

    Rejects:
      * any barrier id used with two DIFFERENT thread counts (e.g. a
        loader-scope (3, 32) from a wait_group fence colliding with a
        consumer-scope (3, 128) reduction sync) — the hardware barrier is a
        counting semaphore per id, so mixed counts wedge the kernel;
      * any id >= 16 or manual-range ids (11..14) besides the allowed set.
    """
    src = kernel.get_kernel_source()
    pats = [
        r"__sync_thread_partial[<(]\s*(\d+)\s*[,:]\s*(\d+)",
        r"__named_barrier_arrive[<(]\s*(\d+)\s*[,:]\s*(\d+)",
    ]
    per_id = {}
    for pat in pats:
        for mid, cnt in re.findall(pat, src):
            per_id.setdefault(int(mid), set()).add(int(cnt))
    bad = {i: sorted(c) for i, c in per_id.items() if len(c) > 1}
    hi = {i: sorted(c) for i, c in per_id.items() if i >= 16 or 11 <= i <= 14}
    if bad:
        raise RuntimeError(f"barrier id used with multiple counts (would wedge): {bad}")
    if hi:
        raise RuntimeError(f"barrier id outside safe range: {hi}")
    return per_id


def _num_sms():
    return get_num_sms()


# ---------------------------------------------------------------------------
# op4 / op5 / op6: GEMV ops on a (ROWS, K) weight; page = (64, 128) tile.
# kind: "o"    x = Attn (bf16),           out X2 = acc + hidden
#       "gu"   x = rmsnorm(X2) (fp32),    out GU  = acc
#       "down" x = silu(g)*u from GU,     out Out = acc + X2
# ---------------------------------------------------------------------------

def _make_ws_gemv(kind: str, rows: int, k_dim: int, seq_len: int):
    NP = NPAGES
    num_sms = _num_sms()
    b = 1
    n_units_total = rows // PR
    dtype = "bfloat16"
    NT = k_dim // PC  # pages per unit

    @tilelang.jit(
        out_idx=None,
        pass_configs={PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    def jit_fn():
        @T.prim_func
        def main(
            W: T.Buffer((rows, k_dim), dtype),
            Hidden: T.Buffer((b, 2048), dtype),
            Attn: T.Buffer((b, 2048), dtype),
            X2: T.Buffer((b, 2048), "float32"),
            GU: T.Buffer((b, 2 * 6144), "float32"),
            Out: T.Buffer((b, 2048), dtype),
        ):
            with T.Kernel(num_sms, threads=THREADS) as bx:
                Pool = T.alloc_shared((NP, PR, PC), dtype)
                sq5 = T.alloc_fragment((2048,), "float32")
                sum5 = T.alloc_fragment((1,), "float32")
                prod = T.alloc_fragment((PR, PC), "float32")
                part = T.alloc_fragment((PR,), "float32")
                acc = T.alloc_fragment((PR,), "float32")

                tid = T.get_thread_binding(0)
                mbars = T.alloc_barrier([NLOAD * 32] * NP + [NCONS * 32] * NP)
                my_units = (n_units_total + num_sms - 1) // num_sms
                gt = T.alloc_local((1,), "int32")
                gt[0] = 0

                if tid < NLOAD * 32:
                    for u in T.serial(my_units):
                        unit = bx * my_units + u
                        if unit < n_units_total:
                            for t in T.serial(NT):
                                g = gt[0] + t
                                p = g % NP
                                if g >= NP:
                                    T.mbarrier_wait_parity(
                                        mbars[NP + p], ((g // NP) - 1) & 1
                                    )
                                T.copy(
                                    W[unit * PR : unit * PR + PR, t * PC : (t + 1) * PC],
                                    Pool[p, :, :],
                                )
                                T.mbarrier_arrive(mbars[p])
                        gt[0] = gt[0] + NT
                else:
                    for u in T.serial(my_units):
                        unit = bx * my_units + u
                        if unit < n_units_total:
                            row0 = unit * PR
                            if kind == "gu":
                                for j in T.Parallel(2048):
                                    sq5[j] = X2[0, j] * X2[0, j]
                                T.reduce_sum(sq5, sum5, dim=0, clear=True)
                                rstd = T.rsqrt(sum5[0] / 2048 + RMS_EPS)
                            T.fill(acc, 0)
                            for t in T.serial(NT):
                                g = gt[0] + t
                                p = g % NP
                                T.mbarrier_wait_parity(mbars[p], (g // NP) & 1)
                                if kind == "o":
                                    for i, j in T.Parallel(PR, PC):
                                        prod[i, j] = Pool[p, i, j].astype(
                                            "float32"
                                        ) * Attn[0, t * PC + j].astype("float32")
                                elif kind == "gu":
                                    for i, j in T.Parallel(PR, PC):
                                        prod[i, j] = Pool[p, i, j].astype(
                                            "float32"
                                        ) * (X2[0, t * PC + j] * rstd)
                                else:  # down: swiGLU folded into the operand
                                    for i, j in T.Parallel(PR, PC):
                                        g6 = GU[0, t * PC + j]
                                        u6 = GU[0, 6144 + t * PC + j]
                                        prod[i, j] = Pool[p, i, j].astype(
                                            "float32"
                                        ) * (g6 / (1.0 + T.exp(-g6)) * u6)
                                T.reduce_sum(prod, part, dim=1, clear=True)
                                for i in T.Parallel(PR):
                                    acc[i] += part[i]
                                T.mbarrier_arrive(mbars[NP + p])
                            if kind == "o":
                                for i in T.Parallel(PR):
                                    X2[0, row0 + i] = (
                                        acc[i] + Hidden[0, row0 + i].astype("float32")
                                    )
                            elif kind == "gu":
                                for i in T.Parallel(PR):
                                    GU[0, row0 + i] = acc[i]
                            else:
                                for i in T.Parallel(PR):
                                    Out[0, row0 + i] = (
                                        acc[i] + X2[0, row0 + i]
                                    ).astype(dtype)
                        gt[0] = gt[0] + NT

        return main

    kern = jit_fn()
    _assert_safe_barrier_ids(kern)
    return kern


# ---------------------------------------------------------------------------
# op1: rmsnorm + qkv GEMV + head norm + KV write-back (unit = one head).
# A (D=128, BK=128) weight tile is two stacked pages (row halves ph=0,1).
# ---------------------------------------------------------------------------

def _make_ws_qkv(hidden: int, q_rows: int, qkv_rows: int, n_heads: int,
                 n_kv_heads: int, head_dim: int, seq_len: int):
    D = head_dim
    NP = NPAGES
    num_sms = _num_sms()
    n_units = qkv_rows // D
    dtype = "bfloat16"
    S = seq_len
    NT = hidden // PC          # K-tiles per unit (BK = 128)
    NPH = D // PR              # pages per K-tile (row halves)

    @tilelang.jit(
        out_idx=None,
        pass_configs={PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    def jit_fn():
        @T.prim_func
        def main(
            Hidden: T.Buffer((1, hidden), dtype),
            Wqkv: T.Buffer((qkv_rows, hidden), dtype),
            Kc: T.Buffer((1, n_kv_heads, S, D), dtype),
            Vc: T.Buffer((1, n_kv_heads, S, D), dtype),
            QKV: T.Buffer((1, qkv_rows), dtype),
        ):
            with T.Kernel(num_sms, threads=THREADS) as bx:
                Pool = T.alloc_shared((NP, PR, PC), dtype)
                prod = T.alloc_fragment((PR, PC), "float32")
                part = T.alloc_fragment((PR,), "float32")
                accA = T.alloc_fragment((PR,), "float32")
                accB = T.alloc_fragment((PR,), "float32")
                sq = T.alloc_fragment((hidden,), "float32")
                sumh = T.alloc_fragment((1,), "float32")
                sqhA = T.alloc_fragment((PR,), "float32")
                sqhB = T.alloc_fragment((PR,), "float32")
                sumq = T.alloc_fragment((1,), "float32")
                sumqB = T.alloc_fragment((1,), "float32")

                tid = T.get_thread_binding(0)
                mbars = T.alloc_barrier([NLOAD * 32] * NP + [NCONS * 32] * NP)
                my_units = (n_units + num_sms - 1) // num_sms
                gt = T.alloc_local((1,), "int32")
                gt[0] = 0

                if tid < NLOAD * 32:
                    for u in T.serial(my_units):
                        head = bx * my_units + u
                        if head < n_units:
                            for t in T.serial(NT):
                                for ph in T.unroll(NPH):
                                    g = gt[0] + t * NPH + ph
                                    p = g % NP
                                    if g >= NP:
                                        T.mbarrier_wait_parity(
                                            mbars[NP + p], ((g // NP) - 1) & 1
                                        )
                                    r0 = head * D + ph * PR
                                    T.copy(
                                        Wqkv[r0 : r0 + PR, t * PC : (t + 1) * PC],
                                        Pool[p, :, :],
                                    )
                                    T.mbarrier_arrive(mbars[p])
                        gt[0] = gt[0] + NT * NPH
                else:
                    for u in T.serial(my_units):
                        head = bx * my_units + u
                        if head < n_units:
                            row0 = head * D
                            for j in T.Parallel(hidden):
                                sq[j] = Hidden[0, j].astype("float32")
                            for j in T.Parallel(hidden):
                                sq[j] = sq[j] * sq[j]
                            T.reduce_sum(sq, sumh, dim=0, clear=True)
                            rstd = T.rsqrt(sumh[0] / hidden + RMS_EPS)
                            T.fill(accA, 0)
                            T.fill(accB, 0)
                            for t in T.serial(NT):
                                for ph in T.unroll(NPH):
                                    g = gt[0] + t * NPH + ph
                                    p = g % NP
                                    T.mbarrier_wait_parity(mbars[p], (g // NP) & 1)
                                    for i, j in T.Parallel(PR, PC):
                                        prod[i, j] = Pool[p, i, j].astype(
                                            "float32"
                                        ) * (
                                            Hidden[0, t * PC + j].astype("float32") * rstd
                                        )
                                    T.reduce_sum(prod, part, dim=1, clear=True)
                                    if ph == 0:
                                        for i in T.Parallel(PR):
                                            accA[i] += part[i]
                                    else:
                                        for i in T.Parallel(PR):
                                            accB[i] += part[i]
                                    T.mbarrier_arrive(mbars[NP + p])
                            # qwen3 q/k head norm on the fp32 accumulator
                            for i in T.Parallel(PR):
                                sqhA[i] = accA[i] * accA[i]
                            for i in T.Parallel(PR):
                                sqhB[i] = accB[i] * accB[i]
                            T.reduce_sum(sqhA, sumq, dim=0, clear=True)
                            T.reduce_sum(sqhB, sumqB, dim=0, clear=True)
                            head_rstd = T.rsqrt(
                                (sumq[0] + sumqB[0]) / D + RMS_EPS
                            )
                            for i in T.Parallel(PR):
                                if head < n_heads + n_kv_heads:
                                    accA[i] = accA[i] * head_rstd
                            for i in T.Parallel(PR):
                                if head < n_heads + n_kv_heads:
                                    accB[i] = accB[i] * head_rstd
                            for i in T.Parallel(PR):
                                QKV[0, row0 + i] = accA[i].astype(dtype)
                            for i in T.Parallel(PR):
                                QKV[0, row0 + PR + i] = accB[i].astype(dtype)
                            if head >= n_heads:
                                if head < n_heads + n_kv_heads:
                                    for i in T.Parallel(PR):
                                        Kc[0, head - n_heads, S - 1, i] = accA[i].astype(dtype)
                                    for i in T.Parallel(PR):
                                        Kc[0, head - n_heads, S - 1, PR + i] = accB[i].astype(dtype)
                            if head >= n_heads + n_kv_heads:
                                for i in T.Parallel(PR):
                                    Vc[0, head - n_heads - n_kv_heads, S - 1, i] = accA[i].astype(dtype)
                                for i in T.Parallel(PR):
                                    Vc[0, head - n_heads - n_kv_heads, S - 1, PR + i] = accB[i].astype(dtype)
                        gt[0] = gt[0] + NT * NPH

        return main

    kern = jit_fn()
    _assert_safe_barrier_ids(kern)
    return kern


# ---------------------------------------------------------------------------
# op2: split flash-decoding attention partials. BS=64 restored: every K/V/Q
# tile is exactly one (64, 128) pool page, staged by the loader.
# ---------------------------------------------------------------------------

def _make_ws_attn(seq_len: int, n_heads: int, n_kv_heads: int, head_dim: int):
    BS, BH = 64, 64
    NP = 4  # + consumer-side Qs (16 KB): Q stays live for the whole unit, so
    # it cannot ride the ring (a late-freed page breaks the FIFO that the
    # loader's FREE waits rely on — found as a unit-0 deadlock).
    num_sms = _num_sms()
    group = n_heads // n_kv_heads
    ns = max(1, min(8, seq_len // 256))
    S = seq_len
    dtype, accum_dtype = "bfloat16", "float32"
    scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    base = (S // ns // BS) * BS
    nb = base // BS
    rem = S - ns * base
    tail_full = rem // BS
    rem_last = rem % BS
    tail_extra = 1 if rem_last > 0 else 0
    NT = nb + tail_full + tail_extra  # per-unit trip count (dummies included)

    @tilelang.jit(
        out_idx=None,
        pass_configs={PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    def jit_fn():
        @T.prim_func
        def main(
            QKV: T.Buffer((1, n_heads * head_dim), dtype),
            Kc: T.Buffer((1, n_kv_heads, S, head_dim), dtype),
            Vc: T.Buffer((1, n_kv_heads, S, head_dim), dtype),
            Opart: T.Buffer((1, n_heads, ns, head_dim), accum_dtype),
            LSE: T.Buffer((1, n_heads, ns), accum_dtype),
        ):
            with T.Kernel(num_sms, threads=THREADS) as bx:
                Pool = T.alloc_shared((NP, BS, PC), dtype)
                Qs = T.alloc_shared((BH, head_dim), dtype)
                Os = T.alloc_shared((group, head_dim), accum_dtype)
                acc_s = T.alloc_fragment((BH, BS), accum_dtype)
                acc_s_cast = T.alloc_fragment((BH, BS), dtype)
                acc_o = T.alloc_fragment((BH, head_dim), accum_dtype)
                scores_max = T.alloc_fragment((BH,), accum_dtype)
                scores_max_prev = T.alloc_fragment((BH,), accum_dtype)
                scores_scale = T.alloc_fragment((BH,), accum_dtype)
                scores_sum = T.alloc_fragment((BH,), accum_dtype)
                logsum = T.alloc_fragment((BH,), accum_dtype)

                tid = T.get_thread_binding(0)
                mbars = T.alloc_barrier([NLOAD * 32] * NP + [NCONS * 32] * NP)
                total2 = n_kv_heads * ns
                my_units = (total2 + num_sms - 1) // num_sms
                gt = T.alloc_local((1,), "int32")
                gt[0] = 0

                if tid < NLOAD * 32:
                    for u in T.serial(my_units):
                        wu = bx * my_units + u
                        if wu < total2:
                            sid = wu % ns
                            hid = (wu // ns) % n_kv_heads
                            for t in T.serial(NT):
                                # K page then V page
                                for kv in T.unroll(2):
                                    g = gt[0] + t * 2 + kv
                                    p = g % NP
                                    if g >= NP:
                                        T.mbarrier_wait_parity(
                                            mbars[NP + p], ((g // NP) - 1) & 1
                                        )
                                    if nb > 0:
                                        src0 = T.if_then_else(
                                            t < nb,
                                            sid * base + t * BS,
                                            T.if_then_else(
                                                sid != ns - 1,
                                                0,  # dummy window, masked to -inf
                                                T.if_then_else(
                                                    t < nb + tail_full,
                                                    ns * base + (t - nb) * BS,
                                                    S - BS,
                                                ),
                                            ),
                                        )
                                        if kv == 0:
                                            T.copy(
                                                Kc[0, hid, src0 : src0 + BS, :],
                                                Pool[p, :, :],
                                            )
                                        else:
                                            T.copy(
                                                Vc[0, hid, src0 : src0 + BS, :],
                                                Pool[p, :, :],
                                            )
                                    else:
                                        # S < BS: guarded flat fill
                                        if kv == 0:
                                            for i, j in T.Parallel(BS, head_dim):
                                                Pool[p, i, j] = T.if_then_else(
                                                    i < S,
                                                    Kc[0, hid, T.min(i, S - 1), j],
                                                    T.cast(0, dtype),
                                                )
                                        else:
                                            for i, j in T.Parallel(BS, head_dim):
                                                Pool[p, i, j] = T.if_then_else(
                                                    i < S,
                                                    Vc[0, hid, T.min(i, S - 1), j],
                                                    T.cast(0, dtype),
                                                )
                                    T.mbarrier_arrive(mbars[p])
                        gt[0] = gt[0] + NT * 2
                else:
                    for u in T.serial(my_units):
                        wu = bx * my_units + u
                        if wu < total2:
                            sid = wu % ns
                            hid = (wu // ns) % n_kv_heads
                            q_row0 = hid * group
                            # stage this tile's valid Q rows, zero-pad rest
                            for i, j in T.Parallel(BH, head_dim):
                                Qs[i, j] = T.if_then_else(
                                    i < group,
                                    QKV[0, T.min(q_row0 + i, n_heads - 1) * head_dim + j],
                                    T.cast(0, dtype),
                                )
                            T.fill(acc_o, 0)
                            T.fill(logsum, 0)
                            T.fill(scores_max, -T.infinity(accum_dtype))
                            for t in T.serial(NT):
                                for kv in T.unroll(2):
                                    g = gt[0] + t * 2 + kv
                                    p = g % NP
                                    T.mbarrier_wait_parity(mbars[p], (g // NP) & 1)
                                    if kv == 0:
                                        T.clear(acc_s)
                                        T.gemm(
                                            Qs,
                                            Pool[p, :, :],
                                            acc_s,
                                            transpose_B=True,
                                            policy=T.GemmWarpPolicy.FullRow,
                                        )
                                    else:
                                        if nb > 0:
                                            if rem > 0:
                                                for i, j in T.Parallel(BH, BS):
                                                    acc_s[i, j] = T.if_then_else(
                                                        t < nb,
                                                        acc_s[i, j],
                                                        T.if_then_else(
                                                            sid == ns - 1,
                                                            T.if_then_else(
                                                                j >= T.if_then_else(
                                                                    t == nb + tail_full,
                                                                    BS - rem_last,
                                                                    0,
                                                                ),
                                                                acc_s[i, j],
                                                                -T.infinity(accum_dtype),
                                                            ),
                                                            -T.infinity(accum_dtype),
                                                        ),
                                                    )
                                        else:
                                            # S < BS: flat-filled tile
                                            for i, j in T.Parallel(BH, BS):
                                                acc_s[i, j] = T.if_then_else(
                                                    j < S,
                                                    acc_s[i, j],
                                                    -T.infinity(accum_dtype),
                                                )
                                        T.copy(scores_max, scores_max_prev)
                                        T.fill(scores_max, -T.infinity(accum_dtype))
                                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                                        for i in T.Parallel(BH):
                                            scores_max[i] = T.max(
                                                scores_max[i], scores_max_prev[i]
                                            )
                                        for i in T.Parallel(BH):
                                            scores_scale[i] = T.exp2(
                                                scores_max_prev[i] * scale
                                                - scores_max[i] * scale
                                            )
                                        for i, j in T.Parallel(BH, BS):
                                            acc_s[i, j] = T.exp2(
                                                acc_s[i, j] * scale - scores_max[i] * scale
                                            )
                                        T.reduce_sum(acc_s, scores_sum, dim=1)
                                        for i in T.Parallel(BH):
                                            logsum[i] = (
                                                logsum[i] * scores_scale[i] + scores_sum[i]
                                            )
                                        T.copy(acc_s, acc_s_cast)
                                        for i, j in T.Parallel(BH, head_dim):
                                            acc_o[i, j] *= scores_scale[i]
                                        T.gemm(
                                            acc_s_cast,
                                            Pool[p, :, :],
                                            acc_o,
                                            policy=T.GemmWarpPolicy.FullRow,
                                        )
                                    T.mbarrier_arrive(mbars[NP + p])
                            for i, j in T.Parallel(BH, head_dim):
                                acc_o[i, j] /= logsum[i]
                            for i in T.Parallel(BH):
                                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                            for i in T.Parallel(BH):
                                if i < group:
                                    LSE[0, q_row0 + i, sid] = logsum[i]
                            T.copy(acc_o[:group, :], Os[:, :])
                            T.copy(
                                Os[:, :],
                                Opart[0, q_row0 : q_row0 + group, sid, :],
                            )
                        gt[0] = gt[0] + NT * 2

        return main

    kern = jit_fn()
    _assert_safe_barrier_ids(kern)
    return kern


# ---------------------------------------------------------------------------
# op3: combine split partials (thread-parallel scalar, no roles)
# ---------------------------------------------------------------------------

def _make_ws_combine(n_heads: int, head_dim: int, seq_len: int):
    ns = max(1, min(8, seq_len // 256))
    num_sms = _num_sms()
    total3 = n_heads * head_dim

    @tilelang.jit(
        out_idx=None,
        pass_configs={PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    def jit_fn():
        @T.prim_func
        def main(
            Opart: T.Buffer((1, n_heads, ns, head_dim), "float32"),
            LSE: T.Buffer((1, n_heads, ns), "float32"),
            Attn: T.Buffer((1, n_heads * head_dim), "bfloat16"),
        ):
            with T.Kernel(num_sms, threads=THREADS) as bx:
                lse_max = T.alloc_local((1,), "float32")
                lse_log = T.alloc_local((1,), "float32")
                o_accum = T.alloc_local((1,), "float32")
                tx = T.get_thread_binding(0)
                stride = num_sms * THREADS
                for it in T.serial((total3 + stride - 1) // stride):
                    idx = it * stride + bx * THREADS + tx
                    if idx < total3:
                        h3 = idx // head_dim
                        d3 = idx % head_dim
                        lse_max[0] = -T.infinity("float32")
                        for s in T.serial(ns):
                            lse_max[0] = T.max(lse_max[0], LSE[0, h3, s])
                        lse_log[0] = 0.0
                        for s in T.serial(ns):
                            lse_log[0] += T.exp2(LSE[0, h3, s] - lse_max[0])
                        lse_log[0] = T.log2(lse_log[0]) + lse_max[0]
                        o_accum[0] = 0.0
                        for s in T.serial(ns):
                            w = T.exp2(LSE[0, h3, s] - lse_log[0])
                            o_accum[0] += Opart[0, h3, s, d3] * w
                        Attn[0, h3 * head_dim + d3] = o_accum[0].astype("bfloat16")

        return main

    kern = jit_fn()
    _assert_safe_barrier_ids(kern)
    return kern
