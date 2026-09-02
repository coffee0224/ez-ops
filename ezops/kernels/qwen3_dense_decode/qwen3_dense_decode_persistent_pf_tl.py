"""Persistent single-launch TileLang backend for the qwen3_dense_decode op,
with L2 prefetch of the next op's data during the software-barrier spin
(decode_ldg.cu port).

Identical kernel structure to the ``tilelang_persistent`` backend — one launch
on a persistent grid of num_sms CTAs, six phases separated by the software
grid barrier (atomic arrival + sense-reversal spin); see that module for the
phase layout and the T.sync_grid() miscompile note.

On top of it, threads that arrive at a barrier early do not just spin: each
spin iteration prefetches one more 128-byte cache line of the next phase's
bulk read into L2, using the same PTX hint as MegaQwen's decode_ldg.cu
(``prefetch.global.L2::evict_last``). The helper is injected via
``T.Kernel(prelude=...)`` and called with ``T.call_extern`` — tilelang has no
native prefetch intrinsic. Lines are strided by global thread id, and each
thread advances a private cursor per iteration, so the prefetched volume
self-limits to the actual wait time; the cursor bound also keeps every offset
expression inside int32.

The only enabled target is barrier 1's Kc/Vc (each capped at L2/2 bytes;
a cache longer than L2 prefetches only a prefix): attention is the one phase
whose operand stream is large and whose predecessor's barrier window is
long — phase 1 runs 32 units on 36 CTAs, so 4 CTAs spin for the whole phase
and every CTA spins out its straggler tail, all while DRAM has headroom.
On RTX 5060 Ti this cuts ~8% off s1k/s4k and ~1-2% at s8k/s16k (where the KV
cache meets or exceeds L2 and only a prefix is pinned), and it also collapses
the run-to-run variance of the baseline (~±5% -> ~±0.5%).

Weight prefetch at the GEMV barriers (Wo/Wgu/Wd) is implemented but disabled
by default (module knobs): those spin windows are sub-µs — the GEMV phases
run a single balanced wave — so there is no window to trade for DRAM fetches
and the prefetch traffic only interferes with the still-running phase
(measured par to -16% end-to-end when all enabled).
"""

import logging
import math

import tilelang
import torch
from tilelang import PassConfigKey
from tilelang import language as T
from tilelang.carver.arch.driver.cuda_driver import get_num_sms

logging.getLogger("tilelang.cache.kernel_cache").setLevel(logging.ERROR)

from ..base_kernel import BaseKernel
from ...registry import register_kernel

RMS_EPS = 1e-6

# Injected before the generated kernel (T.Kernel(prelude=...)). One prefetch
# touches one 128-byte cache line. evict_last pins prefetched lines against
# the streaming phases' normal-priority lines; the plain form (evict_normal)
# leaves replacement policy alone — over-committing evict_last (96 MiB of
# weights through a 32 MiB L2) thrashes instead.
_PF_PRELUDE_LAST = r"""
__device__ __forceinline__ void tl_pf_l2(const void* base, long byte_off) {
    const char* p = reinterpret_cast<const char*>(base) + byte_off;
    asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p));
}
"""
_PF_PRELUDE_NORMAL = r"""
__device__ __forceinline__ void tl_pf_l2(const void* base, long byte_off) {
    const char* p = reinterpret_cast<const char*>(base) + byte_off;
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
"""

# Tuning knobs, read once at kernel-build time (module-level so experiments
# can flip them between builds; the registered backend always uses defaults).
# Measured on RTX 5060 Ti (36 SM, 32 MiB L2): only barrier 1's KV prefetch
# pays off — the GEMV barriers' spin windows are sub-µs (single-wave phases),
# so weight prefetch there is pure interference (all-on: par to -16%;
# KV-only: -8%). evict_last is load-bearing: plain-priority KV lines are
# evicted by the still-running phase-1 Wqkv stream before phase 2 reads
# them, so kv_normal measures *slower* than no prefetch at all. A burst at
# arrival (PF_BURST>0) floods the memory pipe right when the previous
# phase's demand traffic is still active — also a loss.
PF_KV = True  # barrier 1: prefetch Kc/Vc (attention's bulk read)
PF_WO = False  # barriers 2/3: Wo
PF_WGU = False  # barrier 4: Wgu
PF_WD = False  # barrier 5: Wd
PF_EVICT_LAST = True  # evict_last vs plain prefetch hint
PF_BURST = 0  # lines/thread issued immediately on arrival (0 = off)
PF_STRIDE = 1  # spin polls per prefetch line during the wait (>= 1)
PF_L2_CAP = True  # cap each target at L2/2 bytes


@register_kernel("qwen3_dense_decode", "tilelang_persistent_pf")
class Qwen3DenseDecodePersistentPfTileLangKernel(BaseKernel):
    _block_n = 64  # output rows per GEMV work unit (phases 4-6)
    _block_k = 64  # K-tile of every GEMV pipeline
    _block_s = 64  # KV rows per attention tile
    _block_h = 64  # Q rows per attention gemm (tilelang layout inference)
    _num_stages = 2
    _threads = 128

    def __init__(
        self,
        batch: int,
        seq_len: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int | None = None,
    ):
        self.batch = batch
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_dim = head_dim
        if hidden_size & (hidden_size - 1):
            raise ValueError(f"hidden_size ({hidden_size}) must be a power of two")
        if head_dim & (head_dim - 1) or head_dim > 128:
            # phase 1 tiles BN=head_dim rows per unit (one head); the fragment
            # and gemm K-dim budgets assume <= 128.
            raise ValueError(f"head_dim ({head_dim}) must be a power of two <= 128")
        BN, BK, BS = self._block_n, self._block_k, self._block_s
        if hidden_size % BN or intermediate_size % BN or (2 * intermediate_size) % BN:
            raise ValueError(
                f"hidden_size ({hidden_size}) and intermediate_size ({intermediate_size}) "
                f"must be divisible by BLOCK_N={BN}"
            )
        if hidden_size % BK or intermediate_size % BK or num_heads * head_dim % BK:
            raise ValueError(
                f"hidden_size ({hidden_size}), intermediate_size ({intermediate_size}) and "
                f"num_heads * head_dim ({num_heads * head_dim}) must be divisible by BLOCK_K={BK}"
            )
        self.group = num_heads // self.num_kv_heads
        if self.group > self._block_h:
            raise ValueError(
                f"GQA group ({self.group}) must be <= {self._block_h} Q rows per attention tile"
            )
        self.q_rows = num_heads * head_dim
        self.kv_rows = self.num_kv_heads * head_dim
        self.qkv_rows = self.q_rows + 2 * self.kv_rows
        self.sm_scale = 1.0 / math.sqrt(head_dim)
        # kv split count for the flash-decoding phase: enough splits to fill
        # the persistent grid, at least one BLOCK_S-aligned tile per split.
        self.num_split = max(1, min(8, seq_len // 256))
        self.num_sms = get_num_sms()
        self._kernel = None
        self._ws = None  # (qkv, opart, lse, attn, x2, gu, state), allocated lazily once

    def _make_kernel(self):
        B, H, I = self.batch, self.hidden_size, self.intermediate_size
        NH, NKV, D = self.num_heads, self.num_kv_heads, self.head_dim
        S = self.seq_len
        QR, QKVR = self.q_rows, self.qkv_rows
        BN, BK, BS, BH = self._block_n, self._block_k, self._block_s, self._block_h
        NSTAGES, THREADS = self._num_stages, self._threads
        eps, num_sms = RMS_EPS, self.num_sms
        group, head_tiles = self.group, self.num_kv_heads  # one tile per kv head
        ns = self.num_split
        TOTAL_THREADS = num_sms * THREADS
        # attention scale folded with log2(e): exp2(x * scale) == exp(x * sm_scale)
        scale = self.sm_scale * 1.44269504
        # per-split aligned chunk + tail on the last split (attn_decode recipe)
        base = (S // ns // BS) * BS
        nb = base // BS
        rem = S - ns * base
        # The tail rides the same pipelined loop as the aligned tiles: full
        # BLOCK_S tiles copy their exact range, and a partial remainder
        # copies the static window [S-BLOCK_S, S) with the re-read overlap
        # masked to -inf. A K/V access outside the Pipelined loop (flat write
        # or bare copy) trips tilelang's layout inference on the
        # multi-buffered (num_stages, BLOCK_S, D) shape, so S < BLOCK_S
        # (no pipelined loop at all) is the only case using guarded flat
        # loads.
        tail_full = rem // BS
        rem_last = rem % BS
        tail_extra = 1 if rem_last > 0 else 0
        dtype, accum_dtype = "bfloat16", "float32"
        # Prefetch targets (128-byte lines). Each target is capped at half of
        # L2: a matrix larger than that would evict itself (or pin evict_last
        # lines it can't keep), so only a prefix is prefetched instead.
        l2_bytes = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).L2_cache_size
        cap_lines = (l2_bytes // 2) // 128 if PF_L2_CAP else 1 << 30
        pf_kv_lines = min(B * NKV * S * D * 2 // 128, cap_lines)
        pf_wo_lines = min(H * QR * 2 // 128, cap_lines)
        pf_wgu_lines = min((2 * I) * H * 2 // 128, cap_lines)
        pf_wd_lines = min(H * I * 2 // 128, cap_lines)
        pf_prelude = _PF_PRELUDE_LAST if PF_EVICT_LAST else _PF_PRELUDE_NORMAL
        pf_burst, pf_stride = PF_BURST, PF_STRIDE

        @tilelang.jit(
            out_idx=[7],
            pass_configs={PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
        )
        def jit_fn():
            @T.prim_func
            def main(
                Hidden: T.Buffer((B, H), dtype),
                Kc: T.Buffer((B, NKV, S, D), dtype),
                Vc: T.Buffer((B, NKV, S, D), dtype),
                Wqkv: T.Buffer((QKVR, H), dtype),
                Wo: T.Buffer((H, QR), dtype),
                Wgu: T.Buffer((2 * I, H), dtype),
                Wd: T.Buffer((H, I), dtype),
                Out: T.Buffer((B, H), dtype),
                QKV: T.Buffer((B, QKVR), dtype),
                Opart: T.Buffer((B, NH, ns, D), accum_dtype),
                LSE: T.Buffer((B, NH, ns), accum_dtype),
                Attn: T.Buffer((B, QR), dtype),
                X2: T.Buffer((B, H), accum_dtype),
                GU: T.Buffer((B, 2 * I), accum_dtype),
                Counter: T.Buffer((1,), "int32"),
                Sense: T.Buffer((1,), "int32"),
            ):
                with T.Kernel(num_sms, threads=THREADS, prelude=pf_prelude) as bx:
                    # -- phase 1: qkv = rmsnorm(hidden) @ Wqkv.T, head norms,
                    #    k/v write-back; one unit = one head's D rows --------
                    Ws1 = T.alloc_shared((D, BK), dtype)
                    prod1 = T.alloc_fragment((D, BK), "float32")
                    part1 = T.alloc_fragment((D,), "float32")
                    acc1 = T.alloc_fragment((D,), "float32")
                    sq1 = T.alloc_fragment((H,), "float32")
                    sum1 = T.alloc_fragment((1,), "float32")
                    sqh = T.alloc_fragment((D,), "float32")
                    sumh = T.alloc_fragment((1,), "float32")

                    # -- phase 2: split flash-decoding partials ---------------
                    Qs = T.alloc_shared((BH, D), dtype)
                    Ks = T.alloc_shared((BS, D), dtype)
                    Vs = T.alloc_shared((BS, D), dtype)
                    Os = T.alloc_shared((group, D), accum_dtype)
                    acc_s = T.alloc_fragment((BH, BS), accum_dtype)
                    acc_s_cast = T.alloc_fragment((BH, BS), dtype)
                    acc_o = T.alloc_fragment((BH, D), accum_dtype)
                    scores_max = T.alloc_fragment((BH,), accum_dtype)
                    scores_max_prev = T.alloc_fragment((BH,), accum_dtype)
                    scores_scale = T.alloc_fragment((BH,), accum_dtype)
                    scores_sum = T.alloc_fragment((BH,), accum_dtype)
                    logsum = T.alloc_fragment((BH,), accum_dtype)

                    # -- phase 3: combine (thread-parallel scalars) -----------
                    lse_max = T.alloc_local((1,), accum_dtype)
                    lse_log = T.alloc_local((1,), accum_dtype)
                    o_accum = T.alloc_local((1,), accum_dtype)

                    # -- phases 4-6: GEMV fragments ---------------------------
                    Ws4 = T.alloc_shared((BN, BK), dtype)
                    prod4 = T.alloc_fragment((BN, BK), "float32")
                    part4 = T.alloc_fragment((BN,), "float32")
                    acc4 = T.alloc_fragment((BN,), "float32")
                    Xs5 = T.alloc_shared((H,), dtype)
                    Ws5 = T.alloc_shared((BN, BK), dtype)
                    sq5 = T.alloc_fragment((H,), "float32")
                    sum5 = T.alloc_fragment((1,), "float32")
                    prod5 = T.alloc_fragment((BN, BK), "float32")
                    part5 = T.alloc_fragment((BN,), "float32")
                    acc5 = T.alloc_fragment((BN,), "float32")
                    Ws6 = T.alloc_shared((BN, BK), dtype)
                    prod6 = T.alloc_fragment((BN, BK), "float32")
                    part6 = T.alloc_fragment((BN,), "float32")
                    acc6 = T.alloc_fragment((BN,), "float32")
                    my_gen = T.alloc_local((1,), "int32")
                    my_gen[0] = 0
                    # prefetch cursors + global thread id. pf counts a
                    # thread's issued lines (line id = pf*TOTAL_THREADS +
                    # gtid); spin counts poll iterations for PF_STRIDE
                    # throttling. The line guard keeps offsets in int32 and
                    # the prefetched volume tracking the actual wait time.
                    pf = T.alloc_local((1,), "int32")
                    spin = T.alloc_local((1,), "int32")
                    pf[0] = 0
                    spin[0] = 0
                    gtid = bx * THREADS + T.get_thread_binding(0)

                    num_tiles1 = QKVR // D
                    total1 = B * num_tiles1
                    for it in T.serial((total1 + num_sms - 1) // num_sms):
                        wid = it * num_sms + bx
                        if wid < total1:
                            b = wid // num_tiles1
                            head = wid % num_tiles1  # head-row block: [q | k | v]
                            row0 = head * D
                            # rmsnorm statistic of the hidden row (fp32)
                            for j in T.Parallel(H):
                                sq1[j] = Hidden[b, j].astype("float32")
                            for j in T.Parallel(H):
                                sq1[j] = sq1[j] * sq1[j]
                            T.reduce_sum(sq1, sum1, dim=0, clear=True)
                            rstd = T.rsqrt(sum1[0] / H + eps)
                            T.fill(acc1, 0)
                            for k in T.Pipelined(H // BK, num_stages=NSTAGES):
                                T.copy(Wqkv[row0 : row0 + D, k * BK : (k + 1) * BK], Ws1)
                                for i, j in T.Parallel(D, BK):
                                    prod1[i, j] = Ws1[i, j].astype("float32") * (
                                        Hidden[b, k * BK + j].astype("float32") * rstd
                                    )
                                T.reduce_sum(prod1, part1, dim=1, clear=True)
                                for i in T.Parallel(D):
                                    acc1[i] += part1[i]
                            # Qwen3 q/k head-norm over the fp32 accumulator
                            # (v rows skip it).
                            for i in T.Parallel(D):
                                sqh[i] = acc1[i] * acc1[i]
                            T.reduce_sum(sqh, sumh, dim=0, clear=True)
                            head_rstd = T.rsqrt(sumh[0] / D + eps)
                            for i in T.Parallel(D):
                                if head < NH + NKV:
                                    acc1[i] = acc1[i] * head_rstd
                            for i in T.Parallel(D):
                                QKV[b, row0 + i] = acc1[i].astype(dtype)
                            # write the new token's k/v into the cache's last
                            # slot; phase 2 then streams one uniform cache.
                            if head >= NH:
                                if head < NH + NKV:
                                    for i in T.Parallel(D):
                                        Kc[b, head - NH, S - 1, i] = acc1[i].astype(dtype)
                            if head >= NH + NKV:
                                for i in T.Parallel(D):
                                    Vc[b, head - NH - NKV, S - 1, i] = acc1[i].astype(dtype)

                    # software grid barrier (see fused_o_proj_ffn_persistent_tl);
                    # spinners prefetch attention's bulk read — the KV cache.
                    if PF_KV:
                        for ib1 in T.serial(pf_burst):
                            lib1 = ib1 * TOTAL_THREADS + gtid
                            if lib1 < pf_kv_lines:
                                T.evaluate(T.call_extern("void", "tl_pf_l2", Kc.data, lib1 * 128))
                                T.evaluate(T.call_extern("void", "tl_pf_l2", Vc.data, lib1 * 128))
                        pf[0] = pf_burst
                        spin[0] = 0
                        prev1 = T.atomic_add(Counter[0], 1, "acq_rel", True)
                        if prev1 == TOTAL_THREADS - 1:
                            T.atomic_store(Counter[0], 0, "release")
                            T.atomic_add(Sense[0], 1, "acq_rel")
                        else:
                            with T.While(T.atomic_load(Sense[0], "acquire") <= my_gen[0]):
                                spin[0] = spin[0] + 1
                                if spin[0] % pf_stride == 0:
                                    li1 = pf[0] * TOTAL_THREADS + gtid
                                    if li1 < pf_kv_lines:
                                        T.evaluate(T.call_extern("void", "tl_pf_l2", Kc.data, li1 * 128))
                                        T.evaluate(T.call_extern("void", "tl_pf_l2", Vc.data, li1 * 128))
                                        pf[0] = pf[0] + 1
                    else:
                        prev1 = T.atomic_add(Counter[0], 1, "acq_rel", True)
                        if prev1 == TOTAL_THREADS - 1:
                            T.atomic_store(Counter[0], 0, "release")
                            T.atomic_add(Sense[0], 1, "acq_rel")
                        else:
                            with T.While(T.atomic_load(Sense[0], "acquire") <= my_gen[0]):
                                T.evaluate(0)
                    my_gen[0] = my_gen[0] + 1

                    # -- phase 2: split attention partials ---------------------
                    total2 = B * head_tiles * ns
                    for it in T.serial((total2 + num_sms - 1) // num_sms):
                        work_id = it * num_sms + bx
                        if work_id < total2:
                            sid = work_id % ns
                            task = work_id // ns
                            bid = task // head_tiles
                            hid = task % head_tiles
                            q_row0 = hid * group
                            cur_kv_head = hid

                            # Load this tile's valid Q rows from the QKV
                            # workspace, zero-pad the rest (clamped read).
                            for i, j in T.Parallel(BH, D):
                                Qs[i, j] = T.if_then_else(
                                    i < group,
                                    QKV[bid, T.min(q_row0 + i, NH - 1) * D + j],
                                    T.cast(0, dtype),
                                )
                            T.fill(acc_o, 0)
                            T.fill(logsum, 0)
                            T.fill(scores_max, -T.infinity(accum_dtype))

                            if nb > 0:
                                # One pipelined loop covers the split's
                                # aligned tiles plus, on the last split, the
                                # remainder: full tiles at ns*base + m*BS and
                                # one [S-BLOCK_S, S) window whose re-read
                                # overlap with earlier tiles is masked. Every
                                # K/V access is a T.copy inside the same
                                # Pipelined loop — a copy outside it trips
                                # tilelang's layout inference on the
                                # multi-buffered (2, BLOCK_S, D) shape. Other
                                # splits run those extra tiles as dummies:
                                # the stale K/V gemm is masked to -inf, which
                                # contributes nothing (scores_max keeps the
                                # previous, finite maximum).
                                for k in T.Pipelined(nb + tail_full + tail_extra, num_stages=NSTAGES):
                                    if k < nb:
                                        kv_start = sid * base + k * BS
                                        T.copy(Kc[bid, cur_kv_head, kv_start : kv_start + BS, :], Ks)
                                        T.copy(Vc[bid, cur_kv_head, kv_start : kv_start + BS, :], Vs)
                                    else:
                                        if sid == ns - 1:
                                            if k < nb + tail_full:
                                                t0 = ns * base + (k - nb) * BS
                                                T.copy(Kc[bid, cur_kv_head, t0 : t0 + BS, :], Ks)
                                                T.copy(Vc[bid, cur_kv_head, t0 : t0 + BS, :], Vs)
                                            else:
                                                T.copy(Kc[bid, cur_kv_head, S - BS : S, :], Ks)
                                                T.copy(Vc[bid, cur_kv_head, S - BS : S, :], Vs)
                                    T.clear(acc_s)
                                    T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                    if rem > 0:
                                        # window_lo: columns of the window
                                        # tile that re-read already-covered
                                        # positions (full tail tiles use 0).
                                        for i, j in T.Parallel(BH, BS):
                                            acc_s[i, j] = T.if_then_else(
                                                k < nb,
                                                acc_s[i, j],
                                                T.if_then_else(
                                                    sid == ns - 1,
                                                    T.if_then_else(
                                                        j >= T.if_then_else(k == nb + tail_full, BS - rem_last, 0),
                                                        acc_s[i, j],
                                                        -T.infinity(accum_dtype),
                                                    ),
                                                    -T.infinity(accum_dtype),
                                                ),
                                            )
                                    T.copy(scores_max, scores_max_prev)
                                    T.fill(scores_max, -T.infinity(accum_dtype))
                                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                                    for i in T.Parallel(BH):
                                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                                    for i in T.Parallel(BH):
                                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                                    for i, j in T.Parallel(BH, BS):
                                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                                    T.reduce_sum(acc_s, scores_sum, dim=1)
                                    for i in T.Parallel(BH):
                                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                                    T.copy(acc_s, acc_s_cast)
                                    for i, j in T.Parallel(BH, D):
                                        acc_o[i, j] *= scores_scale[i]
                                    T.gemm(acc_s_cast, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)
                            else:
                                # S < BLOCK_S: no pipelined loop exists, so
                                # guarded flat loads cannot conflict with it.
                                if sid == ns - 1:
                                    for i, j in T.Parallel(BS, D):
                                        pos = i
                                        Ks[i, j] = T.if_then_else(
                                            pos < S,
                                            Kc[bid, cur_kv_head, T.min(pos, S - 1), j],
                                            T.cast(0, dtype),
                                        )
                                    for i, j in T.Parallel(BS, D):
                                        pos = i
                                        Vs[i, j] = T.if_then_else(
                                            pos < S,
                                            Vc[bid, cur_kv_head, T.min(pos, S - 1), j],
                                            T.cast(0, dtype),
                                        )
                                T.clear(acc_s)
                                T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
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
                                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                                for i in T.Parallel(BH):
                                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                                for i, j in T.Parallel(BH, BS):
                                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                                T.reduce_sum(acc_s, scores_sum, dim=1)
                                for i in T.Parallel(BH):
                                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                                T.copy(acc_s, acc_s_cast)
                                for i, j in T.Parallel(BH, D):
                                    acc_o[i, j] *= scores_scale[i]
                                T.gemm(acc_s_cast, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)

                            for i, j in T.Parallel(BH, D):
                                acc_o[i, j] /= logsum[i]
                            for i in T.Parallel(BH):
                                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                            for i in T.Parallel(BH):
                                if i < group:
                                    LSE[bid, q_row0 + i, sid] = logsum[i]
                            T.copy(acc_o[:group, :], Os)
                            T.copy(Os, Opart[bid, q_row0 : q_row0 + group, sid, :])

                    # barrier: attention -> combine/o_proj; spinners prefetch Wo
                    # (phase 3 combine reads no weights)
                    if PF_WO:
                        for ib2 in T.serial(pf_burst):
                            lib2 = ib2 * TOTAL_THREADS + gtid
                            if lib2 < pf_wo_lines:
                                T.evaluate(T.call_extern("void", "tl_pf_l2", Wo.data, lib2 * 128))
                        pf[0] = pf_burst
                        spin[0] = 0
                        prev2 = T.atomic_add(Counter[0], 1, "acq_rel", True)
                        if prev2 == TOTAL_THREADS - 1:
                            T.atomic_store(Counter[0], 0, "release")
                            T.atomic_add(Sense[0], 1, "acq_rel")
                        else:
                            with T.While(T.atomic_load(Sense[0], "acquire") <= my_gen[0]):
                                spin[0] = spin[0] + 1
                                if spin[0] % pf_stride == 0:
                                    li2 = pf[0] * TOTAL_THREADS + gtid
                                    if li2 < pf_wo_lines:
                                        T.evaluate(T.call_extern("void", "tl_pf_l2", Wo.data, li2 * 128))
                                        pf[0] = pf[0] + 1
                    else:
                        prev2 = T.atomic_add(Counter[0], 1, "acq_rel", True)
                        if prev2 == TOTAL_THREADS - 1:
                            T.atomic_store(Counter[0], 0, "release")
                            T.atomic_add(Sense[0], 1, "acq_rel")
                        else:
                            with T.While(T.atomic_load(Sense[0], "acquire") <= my_gen[0]):
                                T.evaluate(0)
                    my_gen[0] = my_gen[0] + 1

                    # -- phase 3: combine split partials -> Attn ---------------
                    tx = T.get_thread_binding(0)
                    total3 = B * NH * D
                    stride3 = num_sms * THREADS
                    for it in T.serial((total3 + stride3 - 1) // stride3):
                        idx = it * stride3 + bx * THREADS + tx
                        if idx < total3:
                            b3 = idx // (NH * D)
                            h3 = (idx % (NH * D)) // D
                            d3 = idx % D
                            lse_max[0] = -T.infinity(accum_dtype)
                            for s in T.serial(ns):
                                lse_max[0] = T.max(lse_max[0], LSE[b3, h3, s])
                            lse_log[0] = 0.0
                            for s in T.serial(ns):
                                lse_log[0] += T.exp2(LSE[b3, h3, s] - lse_max[0])
                            lse_log[0] = T.log2(lse_log[0]) + lse_max[0]
                            o_accum[0] = 0.0
                            for s in T.serial(ns):
                                w = T.exp2(LSE[b3, h3, s] - lse_log[0])
                                o_accum[0] += Opart[b3, h3, s, d3] * w
                            Attn[b3, h3 * D + d3] = o_accum[0].astype(dtype)

                    # barrier: combine -> o_proj; keep completing Wo
                    if PF_WO:
                        for ib3 in T.serial(pf_burst):
                            lib3 = ib3 * TOTAL_THREADS + gtid
                            if lib3 < pf_wo_lines:
                                T.evaluate(T.call_extern("void", "tl_pf_l2", Wo.data, lib3 * 128))
                        pf[0] = pf_burst
                        spin[0] = 0
                        prev3 = T.atomic_add(Counter[0], 1, "acq_rel", True)
                        if prev3 == TOTAL_THREADS - 1:
                            T.atomic_store(Counter[0], 0, "release")
                            T.atomic_add(Sense[0], 1, "acq_rel")
                        else:
                            with T.While(T.atomic_load(Sense[0], "acquire") <= my_gen[0]):
                                spin[0] = spin[0] + 1
                                if spin[0] % pf_stride == 0:
                                    li3 = pf[0] * TOTAL_THREADS + gtid
                                    if li3 < pf_wo_lines:
                                        T.evaluate(T.call_extern("void", "tl_pf_l2", Wo.data, li3 * 128))
                                        pf[0] = pf[0] + 1
                    else:
                        prev3 = T.atomic_add(Counter[0], 1, "acq_rel", True)
                        if prev3 == TOTAL_THREADS - 1:
                            T.atomic_store(Counter[0], 0, "release")
                            T.atomic_add(Sense[0], 1, "acq_rel")
                        else:
                            with T.While(T.atomic_load(Sense[0], "acquire") <= my_gen[0]):
                                T.evaluate(0)
                    my_gen[0] = my_gen[0] + 1

                    # -- phase 4: x2 = hidden + attn @ Wo.T --------------------
                    num_tiles4 = H // BN
                    total4 = B * num_tiles4
                    for it in T.serial((total4 + num_sms - 1) // num_sms):
                        wid = it * num_sms + bx
                        if wid < total4:
                            b = wid // num_tiles4
                            row0 = (wid % num_tiles4) * BN
                            T.fill(acc4, 0)
                            for k in T.Pipelined(QR // BK, num_stages=NSTAGES):
                                T.copy(Wo[row0 : row0 + BN, k * BK : (k + 1) * BK], Ws4)
                                for i, j in T.Parallel(BN, BK):
                                    prod4[i, j] = Ws4[i, j].astype("float32") * Attn[
                                        b, k * BK + j
                                    ].astype("float32")
                                T.reduce_sum(prod4, part4, dim=1, clear=True)
                                for i in T.Parallel(BN):
                                    acc4[i] += part4[i]
                            for i in T.Parallel(BN):
                                X2[b, row0 + i] = acc4[i] + Hidden[b, row0 + i].astype("float32")

                    # barrier: o_proj -> gate/up; spinners prefetch Wgu
                    if PF_WGU:
                        for ib4 in T.serial(pf_burst):
                            lib4 = ib4 * TOTAL_THREADS + gtid
                            if lib4 < pf_wgu_lines:
                                T.evaluate(T.call_extern("void", "tl_pf_l2", Wgu.data, lib4 * 128))
                        pf[0] = pf_burst
                        spin[0] = 0
                        prev4 = T.atomic_add(Counter[0], 1, "acq_rel", True)
                        if prev4 == TOTAL_THREADS - 1:
                            T.atomic_store(Counter[0], 0, "release")
                            T.atomic_add(Sense[0], 1, "acq_rel")
                        else:
                            with T.While(T.atomic_load(Sense[0], "acquire") <= my_gen[0]):
                                spin[0] = spin[0] + 1
                                if spin[0] % pf_stride == 0:
                                    li4 = pf[0] * TOTAL_THREADS + gtid
                                    if li4 < pf_wgu_lines:
                                        T.evaluate(T.call_extern("void", "tl_pf_l2", Wgu.data, li4 * 128))
                                        pf[0] = pf[0] + 1
                    else:
                        prev4 = T.atomic_add(Counter[0], 1, "acq_rel", True)
                        if prev4 == TOTAL_THREADS - 1:
                            T.atomic_store(Counter[0], 0, "release")
                            T.atomic_add(Sense[0], 1, "acq_rel")
                        else:
                            with T.While(T.atomic_load(Sense[0], "acquire") <= my_gen[0]):
                                T.evaluate(0)
                    my_gen[0] = my_gen[0] + 1

                    # -- phase 5: gu = rmsnorm(x2) @ Wgu.T ---------------------
                    num_tiles5 = (2 * I) // BN
                    total5 = B * num_tiles5
                    for it in T.serial((total5 + num_sms - 1) // num_sms):
                        wid = it * num_sms + bx
                        if wid < total5:
                            b = wid // num_tiles5
                            row0 = (wid % num_tiles5) * BN
                            for j in T.Parallel(H):
                                Xs5[j] = X2[b, j].astype(dtype)
                            for j in T.Parallel(H):
                                sq5[j] = Xs5[j].astype("float32") * Xs5[j].astype("float32")
                            T.reduce_sum(sq5, sum5, dim=0, clear=True)
                            rstd5 = T.rsqrt(sum5[0] / H + eps)
                            T.fill(acc5, 0)
                            for k in T.Pipelined(H // BK, num_stages=NSTAGES):
                                T.copy(Wgu[row0 : row0 + BN, k * BK : (k + 1) * BK], Ws5)
                                for i, j in T.Parallel(BN, BK):
                                    prod5[i, j] = Ws5[i, j].astype("float32") * (
                                        Xs5[k * BK + j].astype("float32") * rstd5
                                    )
                                T.reduce_sum(prod5, part5, dim=1, clear=True)
                                for i in T.Parallel(BN):
                                    acc5[i] += part5[i]
                            for i in T.Parallel(BN):
                                GU[b, row0 + i] = acc5[i]

                    # barrier: gate/up -> down; spinners prefetch Wd
                    if PF_WD:
                        for ib5 in T.serial(pf_burst):
                            lib5 = ib5 * TOTAL_THREADS + gtid
                            if lib5 < pf_wd_lines:
                                T.evaluate(T.call_extern("void", "tl_pf_l2", Wd.data, lib5 * 128))
                        pf[0] = pf_burst
                        spin[0] = 0
                        prev5 = T.atomic_add(Counter[0], 1, "acq_rel", True)
                        if prev5 == TOTAL_THREADS - 1:
                            T.atomic_store(Counter[0], 0, "release")
                            T.atomic_add(Sense[0], 1, "acq_rel")
                        else:
                            with T.While(T.atomic_load(Sense[0], "acquire") <= my_gen[0]):
                                spin[0] = spin[0] + 1
                                if spin[0] % pf_stride == 0:
                                    li5 = pf[0] * TOTAL_THREADS + gtid
                                    if li5 < pf_wd_lines:
                                        T.evaluate(T.call_extern("void", "tl_pf_l2", Wd.data, li5 * 128))
                                        pf[0] = pf[0] + 1
                    else:
                        prev5 = T.atomic_add(Counter[0], 1, "acq_rel", True)
                        if prev5 == TOTAL_THREADS - 1:
                            T.atomic_store(Counter[0], 0, "release")
                            T.atomic_add(Sense[0], 1, "acq_rel")
                        else:
                            with T.While(T.atomic_load(Sense[0], "acquire") <= my_gen[0]):
                                T.evaluate(0)
                    my_gen[0] = my_gen[0] + 1

                    # -- phase 6: out = x2 + (silu(g)*u) @ Wd.T ----------------
                    num_tiles6 = H // BN
                    total6 = B * num_tiles6
                    for it in T.serial((total6 + num_sms - 1) // num_sms):
                        wid = it * num_sms + bx
                        if wid < total6:
                            b = wid // num_tiles6
                            row0 = (wid % num_tiles6) * BN
                            T.fill(acc6, 0)
                            for k in T.Pipelined(I // BK, num_stages=NSTAGES):
                                T.copy(Wd[row0 : row0 + BN, k * BK : (k + 1) * BK], Ws6)
                                # SwiGLU folded into the operand read: g/u are
                                # the two halves of the GU row, silu(g)*u is
                                # recomputed per K-tile in fp32.
                                for i, j in T.Parallel(BN, BK):
                                    prod6[i, j] = Ws6[i, j].astype("float32") * (
                                        GU[b, k * BK + j]
                                        / (1.0 + T.exp(-GU[b, k * BK + j]))
                                        * GU[b, I + k * BK + j]
                                    )
                                T.reduce_sum(prod6, part6, dim=1, clear=True)
                                for i in T.Parallel(BN):
                                    acc6[i] += part6[i]
                            for i in T.Parallel(BN):
                                Out[b, row0 + i] = (acc6[i] + X2[b, row0 + i]).astype(dtype)

            return main

        return jit_fn()

    def __call__(
        self,
        hidden: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        Wqkv: torch.Tensor,
        Wo: torch.Tensor,
        Wgu: torch.Tensor,
        Wd: torch.Tensor,
    ) -> torch.Tensor:
        B, H, I = self.batch, self.hidden_size, self.intermediate_size
        assert hidden.is_cuda and hidden.dtype == torch.bfloat16
        assert hidden.shape == (B, H) and hidden.is_contiguous()
        assert K_cache.shape == (B, self.num_kv_heads, self.seq_len, self.head_dim)
        assert K_cache.is_contiguous() and V_cache.is_contiguous()
        assert V_cache.shape == K_cache.shape
        assert Wqkv.shape == (self.qkv_rows, H) and Wqkv.is_contiguous()
        assert Wo.shape == (H, self.q_rows) and Wo.is_contiguous()
        assert Wgu.shape == (2 * I, H) and Wgu.is_contiguous()
        assert Wd.shape == (H, I) and Wd.is_contiguous()

        if self._kernel is None:
            self._kernel = self._make_kernel()
            dev = hidden.device
            self._ws = (
                torch.empty((B, self.qkv_rows), device=dev, dtype=torch.bfloat16),  # QKV
                torch.empty(
                    (B, self.num_heads, self.num_split, self.head_dim),
                    device=dev,
                    dtype=torch.float32,
                ),  # Opart
                torch.empty(
                    (B, self.num_heads, self.num_split), device=dev, dtype=torch.float32
                ),  # LSE
                torch.empty((B, self.q_rows), device=dev, dtype=torch.bfloat16),  # Attn
                torch.empty((B, H), device=dev, dtype=torch.float32),  # X2
                torch.empty((B, 2 * I), device=dev, dtype=torch.float32),  # GU
                torch.zeros(2, device=dev, dtype=torch.int32),  # [Counter, Sense]
            )
        qkv, opart, lse, attn, x2, gu, state = self._ws
        # Deterministic barrier state for this launch (stream-ordered memset).
        state.zero_()
        counter, sense = state[:1], state[1:]
        return self._kernel(hidden, K_cache, V_cache, Wqkv, Wo, Wgu, Wd, qkv, opart, lse, attn, x2, gu, counter, sense)
