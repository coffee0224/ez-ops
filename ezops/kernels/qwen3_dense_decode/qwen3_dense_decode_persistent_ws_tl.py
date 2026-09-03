"""Persistent warp-specialized TileLang backend for qwen3_dense_decode (V2).

Megakernel form mirroring pdl-megakernel-reconstruction: ONE kernel launch,
all six ops share a single shared-memory page pool, dependencies are checked
at consume points (scoreboard), and there are NO grid barriers — the loader
free-runs ahead across op boundaries bounded only by pool depth.

Layout (persistent grid of num_sms CTAs x 160 threads = 1 loader warp +
4 consumer warps):

  * page pool: Pool[NP, 64, 128] bf16 pages (16 KB each) shared by every op
    (gemv weight tiles, qkv row-half tiles, attention K/V tiles). One global
    mbarrier ring over the pages, walked in global tile order g by BOTH
    roles:  loader waits FREE[g%NP] parity ((g//NP)-1)&1 before overwriting,
    consumer waits READY[g%NP] parity (g//NP)&1 before reading, and each
    side arrives the matching barrier after fill/consume. Hardware phase
    bits make cross-op page reuse drift-free (stress-validated on odd NP).
  * loader: plain T.copy only. T.async_copy + ptx_wait_group would emit a
    loader-scope (3, 32) partial barrier that collides with consumer (3, 128)
    syncs — the ThreadSync passes assign ids per scope and both start at 3;
    that collision was the V1 intermittent deadlock (now asserted against).
    A T.copy loader still streams at ~86% of HBM peak.
  * op queue Queue[num_sms, 6, 4] = (opcode, start, end, pad): both roles
    walk it and dispatch per-op bodies; the queue, not the kernel text,
    decides what runs.
  * scoreboard Bar[7, 40] gmem counters (g.Bar style), fine-grained:
      op1 -> per head        Bar[1, head]           (target 1)
      op2 -> per q head      Bar[2, q_head]         (target ns)
      op3 -> per CTA         Bar[3, 0]              (target num_sms)
      op4 -> per unit        Bar[4, 0]              (target B*H/64)
      op5 -> per unit        Bar[5, 0]              (target B*2I/64)
    Consumers spin the counters they read (attention's Q rows, combine's
    per-head partials, the GEMV whole-op vectors); the LOADER spins op1's
    k/v writeback counters before filling attention K/V pages (the [S-BS,S)
    window re-reads the token op1 just appended). Every producer does a
    named-barrier store fence (id 15, 128 threads) before its release-add so
    all 128 consumers' gmem stores are visible to the acquirer.
  * attention: BS=64 restored (K/V tiles are single pool pages); Q rides a
    consumer-side Qs buffer because it stays live for the whole unit — a
    late-freed page would break the ring's FIFO order.

Because the loader's only dependency is the op1 k/v writeback, it prefetches
op N+1's weight pages while consumers still compute op N: cross-op overlap
comes from the pool depth, not from barriers.
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
from .qwen3_dense_decode_ws_tl import (
    NCONS,
    NCONS_THREADS,
    NLOAD,
    PR,
    PC,
    THREADS,
    _assert_safe_barrier_ids,
)

RMS_EPS = 1e-6

NP = 4  # pool pages; + Qs(16KB) + Os fits the ~99KB dynamic smem budget

OP_QKV = 1
OP_ATTN = 2
OP_COMBINE = 3
OP_O = 4
OP_GU = 5
OP_DOWN = 6

STORE_FENCE = 15  # named barrier: consumer store fence before scoreboard adds


@register_kernel("qwen3_dense_decode", "tilelang_persistent_ws")
class Qwen3DenseDecodePersistentWsTileLangKernel(BaseKernel):
    _block_n = 64
    _block_k = 64
    _block_s = 64
    _block_h = 64

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
        if batch != 1:
            # The decode plan (op queue, scoreboard targets, [0, b]-style
            # activation indexing) is built for single-batch decode.
            raise ValueError(f"batch must be 1, got {batch}")
        if hidden_size & (hidden_size - 1):
            raise ValueError(f"hidden_size ({hidden_size}) must be a power of two")
        if head_dim & (head_dim - 1) or head_dim > 128:
            raise ValueError(f"head_dim ({head_dim}) must be a power of two <= 128")
        BN, BK, BS = self._block_n, self._block_k, self._block_s
        if hidden_size % BN or intermediate_size % BN or (2 * intermediate_size) % BN:
            raise ValueError(
                f"hidden_size ({hidden_size}) and intermediate_size ({intermediate_size}) "
                f"must be divisible by BLOCK_N={BN}"
            )
        if hidden_size % PC or intermediate_size % PC or num_heads * head_dim % PC:
            raise ValueError(
                f"hidden_size ({hidden_size}), intermediate_size ({intermediate_size}) and "
                f"num_heads * head_dim ({num_heads * head_dim}) must be divisible by "
                f"the page width {PC}"
            )
        if head_dim != PR + PR and head_dim % PR:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by {PR}")
        self.group = num_heads // self.num_kv_heads
        if num_heads % self.num_kv_heads:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads "
                f"({self.num_kv_heads})"
            )
        if self.group > self._block_h:
            raise ValueError(
                f"GQA group ({self.group}) must be <= {self._block_h} Q rows per attention tile"
            )
        self.q_rows = num_heads * head_dim
        self.kv_rows = self.num_kv_heads * head_dim
        self.qkv_rows = self.q_rows + 2 * self.kv_rows
        self.sm_scale = 1.0 / math.sqrt(head_dim)
        self.num_split = max(1, min(8, seq_len // 256))
        self.num_sms = get_num_sms()
        self._kernel = None
        self._ws = None

    # -- host-side op queue (megakernel scheduler, fixed plan) ---------------

    def _build_queue(self) -> torch.Tensor:
        """One queue per SM: [op, start, end, 0] rows for ops 1..6.

        Unit ranges are contiguous slices of each op's flat work list (like
        megakernels' schedule_qkv). The decode plan is static per shape, so
        the queue is built once; only the scoreboard is re-zeroed per call.
        """
        ns, nsm = self.num_split, self.num_sms
        B, H, I = self.batch, self.hidden_size, self.intermediate_size
        NKV, D = self.num_kv_heads, self.head_dim
        BN = self._block_n
        totals = {
            OP_QKV: B * (self.qkv_rows // D),
            OP_ATTN: B * NKV * ns,
            OP_COMBINE: B * self.num_heads * D,
            OP_O: B * (H // BN),
            OP_GU: B * ((2 * I) // BN),
            OP_DOWN: B * (H // BN),
        }
        q = torch.zeros(nsm, 6, 4, dtype=torch.int32)
        for op, total in totals.items():
            per = (total + nsm - 1) // nsm
            for sm in range(nsm):
                q[sm, op - 1, 0] = op
                q[sm, op - 1, 1] = min(sm * per, total)
                q[sm, op - 1, 2] = min((sm + 1) * per, total)
        return q

    def _make_kernel(self):
        B, H, I = self.batch, self.hidden_size, self.intermediate_size
        NH, NKV, D = self.num_heads, self.num_kv_heads, self.head_dim
        S = self.seq_len
        QR, QKVR = self.q_rows, self.qkv_rows
        eps, num_sms = RMS_EPS, self.num_sms
        group = self.group
        ns = self.num_split
        scale = self.sm_scale * 1.44269504
        base = (S // ns // PR) * PR   # per-split aligned chunk (BS = PR = 64)
        nb = base // PR
        rem = S - ns * base
        tail_full = rem // PR
        rem_last = rem % PR
        tail_extra = 1 if rem_last > 0 else 0
        NT_ATTN = nb + tail_full + tail_extra
        dtype, accum_dtype = "bfloat16", "float32"
        BS, BH = PR, PR  # attention tile = one pool page; Q rows per gemm
        total3 = B * NH * D
        total4 = B * (H // PR)
        total5 = B * ((2 * I) // PR)
        NT1 = (H // PC) * (D // PR)   # qkv pages per unit (tile x row-halves)
        NT4 = QR // PC                # o_proj pages per unit
        NT5 = H // PC                 # gate_up pages per unit
        NT6 = I // PC                 # down pages per unit
        # per-batch unit counts for decoding (b, unit) from flat indices
        T1 = QKVR // D   # qkv heads
        TA = NKV * ns    # attention units per batch
        T4 = H // PR     # o/down units per batch
        T5 = (2 * I) // PR  # gate_up units per batch

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
                Queue: T.Buffer((num_sms, 6, 4), "int32"),
                Bar: T.Buffer((7, 40), "int32"),
            ):
                with T.Kernel(num_sms, threads=THREADS) as bx:
                    # -- the one shared page pool --------------------------
                    Pool = T.alloc_shared((NP, PR, PC), dtype)
                    # -- attention state (Q lives for a whole unit: it cannot
                    #    ride the ring without breaking the FREE FIFO) ------
                    Qs = T.alloc_shared((BH, D), dtype)
                    Os = T.alloc_shared((group, D), accum_dtype)
                    acc_s = T.alloc_fragment((BH, BS), accum_dtype)
                    acc_s_cast = T.alloc_fragment((BH, BS), dtype)
                    acc_o = T.alloc_fragment((BH, D), accum_dtype)
                    scores_max = T.alloc_fragment((BH,), accum_dtype)
                    scores_max_prev = T.alloc_fragment((BH,), accum_dtype)
                    scores_scale = T.alloc_fragment((BH,), accum_dtype)
                    scores_sum = T.alloc_fragment((BH,), accum_dtype)
                    logsum = T.alloc_fragment((BH,), accum_dtype)
                    # -- qkv state ------------------------------------------
                    prod1 = T.alloc_fragment((PR, PC), "float32")
                    part1 = T.alloc_fragment((PR,), "float32")
                    accA = T.alloc_fragment((PR,), "float32")
                    accB = T.alloc_fragment((PR,), "float32")
                    sq1 = T.alloc_fragment((H,), "float32")
                    sum1 = T.alloc_fragment((1,), "float32")
                    sqhA = T.alloc_fragment((PR,), "float32")
                    sqhB = T.alloc_fragment((PR,), "float32")
                    sum1B = T.alloc_fragment((1,), "float32")
                    sumqB = T.alloc_fragment((1,), "float32")
                    # -- gemv state ------------------------------------------
                    prod = T.alloc_fragment((PR, PC), "float32")
                    part = T.alloc_fragment((PR,), "float32")
                    acc = T.alloc_fragment((PR,), "float32")
                    sq5 = T.alloc_fragment((H,), "float32")
                    sum5 = T.alloc_fragment((1,), "float32")
                    # -- combine state ---------------------------------------
                    lse_max = T.alloc_local((1,), accum_dtype)
                    lse_log = T.alloc_local((1,), accum_dtype)
                    o_accum = T.alloc_local((1,), accum_dtype)

                    tid = T.get_thread_binding(0)
                    mbars = T.alloc_barrier([NLOAD * 32] * NP + [NCONS * 32] * NP)
                    gt = T.alloc_local((1,), "int32")
                    gt[0] = 0

                    if tid < NLOAD * 32:
                        # ================= LOADER: walk the op queue ========
                        for qi in T.serial(6):
                            op = Queue[bx, qi, 0]
                            st = Queue[bx, qi, 1]
                            en = Queue[bx, qi, 2]
                            if op == OP_QKV:
                                for u in T.serial(en - st):
                                    head = st + u
                                    for t in T.serial(H // PC):
                                        for ph in T.unroll(D // PR):
                                            g = gt[0] + t * (D // PR) + ph
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
                                gt[0] = gt[0] + (en - st) * NT1
                            if op == OP_ATTN:
                                for u in T.serial(en - st):
                                    wu = st + u
                                    sid = wu % ns
                                    hid = (wu // ns) % NKV
                                    # K/V freshness: the [S-BS,S) window re-reads
                                    # the token op1 just appended to the cache.
                                    with T.While(
                                        T.atomic_load(Bar[1, NH + hid], "acquire") < 1
                                    ):
                                        T.evaluate(0)
                                    with T.While(
                                        T.atomic_load(
                                            Bar[1, NH + NKV + hid], "acquire"
                                        )
                                        < B
                                    ):
                                        T.evaluate(0)
                                    for t in T.serial(NT_ATTN):
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
                                                        0,  # dummy, masked -inf
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
                                                    for i, j in T.Parallel(BS, D):
                                                        Pool[p, i, j] = T.if_then_else(
                                                            i < S,
                                                            Kc[0, hid, T.min(i, S - 1), j],
                                                            T.cast(0, dtype),
                                                        )
                                                else:
                                                    for i, j in T.Parallel(BS, D):
                                                        Pool[p, i, j] = T.if_then_else(
                                                            i < S,
                                                            Vc[0, hid, T.min(i, S - 1), j],
                                                            T.cast(0, dtype),
                                                        )
                                            T.mbarrier_arrive(mbars[p])
                                gt[0] = gt[0] + (en - st) * NT_ATTN * 2
                            if op == OP_O:
                                for u in T.serial(en - st):
                                    row0 = (st + u) * PR
                                    for t in T.serial(NT4):
                                        g = gt[0] + t
                                        p = g % NP
                                        if g >= NP:
                                            T.mbarrier_wait_parity(
                                                mbars[NP + p], ((g // NP) - 1) & 1
                                            )
                                        T.copy(
                                            Wo[row0 : row0 + PR, t * PC : (t + 1) * PC],
                                            Pool[p, :, :],
                                        )
                                        T.mbarrier_arrive(mbars[p])
                                gt[0] = gt[0] + (en - st) * NT4
                            if op == OP_GU:
                                for u in T.serial(en - st):
                                    row0 = (st + u) * PR
                                    for t in T.serial(NT5):
                                        g = gt[0] + t
                                        p = g % NP
                                        if g >= NP:
                                            T.mbarrier_wait_parity(
                                                mbars[NP + p], ((g // NP) - 1) & 1
                                            )
                                        T.copy(
                                            Wgu[row0 : row0 + PR, t * PC : (t + 1) * PC],
                                            Pool[p, :, :],
                                        )
                                        T.mbarrier_arrive(mbars[p])
                                gt[0] = gt[0] + (en - st) * NT5
                            if op == OP_DOWN:
                                for u in T.serial(en - st):
                                    row0 = (st + u) * PR
                                    for t in T.serial(NT6):
                                        g = gt[0] + t
                                        p = g % NP
                                        if g >= NP:
                                            T.mbarrier_wait_parity(
                                                mbars[NP + p], ((g // NP) - 1) & 1
                                            )
                                        T.copy(
                                            Wd[row0 : row0 + PR, t * PC : (t + 1) * PC],
                                            Pool[p, :, :],
                                        )
                                        T.mbarrier_arrive(mbars[p])
                                gt[0] = gt[0] + (en - st) * NT6
                    else:
                        # ================= CONSUMER: walk the op queue =====
                        ct = tid - NLOAD * 32
                        for qi in T.serial(6):
                            op = Queue[bx, qi, 0]
                            st = Queue[bx, qi, 1]
                            en = Queue[bx, qi, 2]
                            if op == OP_QKV:
                                for u in T.serial(en - st):
                                    head = st + u
                                    row0 = head * D
                                    for j in T.Parallel(H):
                                        sq1[j] = Hidden[0, j].astype("float32")
                                    for j in T.Parallel(H):
                                        sq1[j] = sq1[j] * sq1[j]
                                    T.reduce_sum(sq1, sum1, dim=0, clear=True)
                                    rstd = T.rsqrt(sum1[0] / H + eps)
                                    T.fill(accA, 0)
                                    T.fill(accB, 0)
                                    for t in T.serial(H // PC):
                                        for ph in T.unroll(D // PR):
                                            g = gt[0] + t * (D // PR) + ph
                                            p = g % NP
                                            T.mbarrier_wait_parity(
                                                mbars[p], (g // NP) & 1
                                            )
                                            for i, j in T.Parallel(PR, PC):
                                                prod1[i, j] = Pool[p, i, j].astype(
                                                    "float32"
                                                ) * (
                                                    Hidden[0, t * PC + j].astype("float32") * rstd
                                                )
                                            T.reduce_sum(prod1, part1, dim=1, clear=True)
                                            if ph == 0:
                                                for i in T.Parallel(PR):
                                                    accA[i] += part1[i]
                                            else:
                                                for i in T.Parallel(PR):
                                                    accB[i] += part1[i]
                                            T.mbarrier_arrive(mbars[NP + p])
                                    # qwen3 q/k head norm on the fp32 accumulator
                                    for i in T.Parallel(PR):
                                        sqhA[i] = accA[i] * accA[i]
                                    for i in T.Parallel(PR):
                                        sqhB[i] = accB[i] * accB[i]
                                    T.reduce_sum(sqhA, sum1, dim=0, clear=True)
                                    T.reduce_sum(sqhB, sum1B, dim=0, clear=True)
                                    head_rstd = T.rsqrt(
                                        (sum1[0] + sum1B[0]) / D + eps
                                    )
                                    for i in T.Parallel(PR):
                                        if head < NH + NKV:
                                            accA[i] = accA[i] * head_rstd
                                    for i in T.Parallel(PR):
                                        if head < NH + NKV:
                                            accB[i] = accB[i] * head_rstd
                                    for i in T.Parallel(PR):
                                        QKV[0, row0 + i] = accA[i].astype(dtype)
                                    for i in T.Parallel(PR):
                                        QKV[0, row0 + PR + i] = accB[i].astype(dtype)
                                    if head >= NH:
                                        if head < NH + NKV:
                                            for i in T.Parallel(PR):
                                                Kc[0, head - NH, S - 1, i] = accA[i].astype(dtype)
                                            for i in T.Parallel(PR):
                                                Kc[0, head - NH, S - 1, PR + i] = accB[i].astype(dtype)
                                    if head >= NH + NKV:
                                        for i in T.Parallel(PR):
                                            Vc[0, head - NH - NKV, S - 1, i] = accA[i].astype(dtype)
                                        for i in T.Parallel(PR):
                                            Vc[0, head - NH - NKV, S - 1, PR + i] = accB[i].astype(dtype)
                                    T.sync_threads(STORE_FENCE, NCONS_THREADS)
                                    if ct == 0:
                                        T.atomic_add(Bar[1, head], 1, "release")
                                gt[0] = gt[0] + (en - st) * NT1
                            if op == OP_ATTN:
                                for u in T.serial(en - st):
                                    wu = st + u
                                    sid = wu % ns
                                    hid = (wu // ns) % NKV
                                    q_row0 = hid * group
                                    # Q freshness for this tile's q rows
                                    for i in T.unroll(group):
                                        with T.While(
                                            T.atomic_load(Bar[1, q_row0 + i], "acquire") < 1
                                        ):
                                            T.evaluate(0)
                                    # stage valid Q rows, zero-pad the rest
                                    for i, j in T.Parallel(BH, D):
                                        Qs[i, j] = T.if_then_else(
                                            i < group,
                                            QKV[0, T.min(q_row0 + i, NH - 1) * D + j],
                                            T.cast(0, dtype),
                                        )
                                    T.fill(acc_o, 0)
                                    T.fill(logsum, 0)
                                    T.fill(scores_max, -T.infinity(accum_dtype))
                                    for t in T.serial(NT_ATTN):
                                        for kv in T.unroll(2):
                                            g = gt[0] + t * 2 + kv
                                            p = g % NP
                                            T.mbarrier_wait_parity(
                                                mbars[p], (g // NP) & 1
                                            )
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
                                                    # S < BS: flat-filled tile,
                                                    # mask the zero padding
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
                                                for i, j in T.Parallel(BH, D):
                                                    acc_o[i, j] *= scores_scale[i]
                                                T.gemm(
                                                    acc_s_cast,
                                                    Pool[p, :, :],
                                                    acc_o,
                                                    policy=T.GemmWarpPolicy.FullRow,
                                                )
                                            T.mbarrier_arrive(mbars[NP + p])
                                    for i, j in T.Parallel(BH, D):
                                        acc_o[i, j] /= logsum[i]
                                    for i in T.Parallel(BH):
                                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                                    for i in T.Parallel(BH):
                                        if i < group:
                                            LSE[0, q_row0 + i, sid] = logsum[i]
                                    T.copy(acc_o[:group, :], Os[:, :])
                                    T.copy(Os[:, :], Opart[0, q_row0 : q_row0 + group, sid, :])
                                    T.sync_threads(STORE_FENCE, NCONS_THREADS)
                                    if ct == 0:
                                        for i in T.unroll(group):
                                            T.atomic_add(Bar[2, q_row0 + i], 1, "release")
                                gt[0] = gt[0] + (en - st) * NT_ATTN * 2
                            if op == OP_COMBINE:
                                for k in T.serial((en - st + NCONS_THREADS - 1) // NCONS_THREADS):
                                    e = st + k * NCONS_THREADS + ct
                                    if e < en:
                                        h3 = e // D
                                        d3 = e % D
                                        with T.While(
                                            T.atomic_load(Bar[2, h3], "acquire") < 1 * ns
                                        ):
                                            T.evaluate(0)
                                        lse_max[0] = -T.infinity(accum_dtype)
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
                                        Attn[0, h3 * D + d3] = o_accum[0].astype(dtype)
                                T.sync_threads(STORE_FENCE, NCONS_THREADS)
                                if ct == 0:
                                    T.atomic_add(Bar[3, 0], 1, "release")
                            if op == OP_O:
                                with T.While(T.atomic_load(Bar[3, 0], "acquire") < num_sms):
                                    T.evaluate(0)
                                for u in T.serial(en - st):
                                    row0 = (st + u) * PR
                                    T.fill(acc, 0)
                                    for t in T.serial(NT4):
                                        g = gt[0] + t
                                        p = g % NP
                                        T.mbarrier_wait_parity(mbars[p], (g // NP) & 1)
                                        for i, j in T.Parallel(PR, PC):
                                            prod[i, j] = Pool[p, i, j].astype("float32") * Attn[
                                                0, t * PC + j
                                            ].astype("float32")
                                        T.reduce_sum(prod, part, dim=1, clear=True)
                                        for i in T.Parallel(PR):
                                            acc[i] += part[i]
                                        T.mbarrier_arrive(mbars[NP + p])
                                    for i in T.Parallel(PR):
                                        X2[0, row0 + i] = acc[i] + Hidden[0, row0 + i].astype(
                                            "float32"
                                        )
                                    T.sync_threads(STORE_FENCE, NCONS_THREADS)
                                    if ct == 0:
                                        T.atomic_add(Bar[4, 0], 1, "release")
                                gt[0] = gt[0] + (en - st) * NT4
                            if op == OP_GU:
                                with T.While(T.atomic_load(Bar[4, 0], "acquire") < total4):
                                    T.evaluate(0)
                                # rmsnorm stats computed ONCE per CTA: B = 1,
                                # so every unit of this op reads the same X2 row.
                                for j in T.Parallel(H):
                                    sq5[j] = X2[0, j] * X2[0, j]
                                T.reduce_sum(sq5, sum5, dim=0, clear=True)
                                rstd5 = T.rsqrt(sum5[0] / H + eps)
                                for u in T.serial(en - st):
                                    row0 = (st + u) * PR
                                    T.fill(acc, 0)
                                    for t in T.serial(NT5):
                                        g = gt[0] + t
                                        p = g % NP
                                        T.mbarrier_wait_parity(mbars[p], (g // NP) & 1)
                                        for i, j in T.Parallel(PR, PC):
                                            prod[i, j] = Pool[p, i, j].astype("float32") * (
                                                X2[0, t * PC + j] * rstd5
                                            )
                                        T.reduce_sum(prod, part, dim=1, clear=True)
                                        for i in T.Parallel(PR):
                                            acc[i] += part[i]
                                        T.mbarrier_arrive(mbars[NP + p])
                                    for i in T.Parallel(PR):
                                        GU[0, row0 + i] = acc[i]
                                    T.sync_threads(STORE_FENCE, NCONS_THREADS)
                                    if ct == 0:
                                        T.atomic_add(Bar[5, 0], 1, "release")
                                gt[0] = gt[0] + (en - st) * NT5
                            if op == OP_DOWN:
                                with T.While(T.atomic_load(Bar[5, 0], "acquire") < total5):
                                    T.evaluate(0)
                                for u in T.serial(en - st):
                                    row0 = (st + u) * PR
                                    T.fill(acc, 0)
                                    for t in T.serial(NT6):
                                        g = gt[0] + t
                                        p = g % NP
                                        T.mbarrier_wait_parity(mbars[p], (g // NP) & 1)
                                        for i, j in T.Parallel(PR, PC):
                                            g6 = GU[0, t * PC + j]
                                            u6 = GU[0, I + t * PC + j]
                                            prod[i, j] = Pool[p, i, j].astype("float32") * (
                                                g6 / (1.0 + T.exp(-g6)) * u6
                                            )
                                        T.reduce_sum(prod, part, dim=1, clear=True)
                                        for i in T.Parallel(PR):
                                            acc[i] += part[i]
                                        T.mbarrier_arrive(mbars[NP + p])
                                    for i in T.Parallel(PR):
                                        Out[0, row0 + i] = (acc[i] + X2[0, row0 + i]).astype(dtype)

            return main

        kern = jit_fn()
        _assert_safe_barrier_ids(kern)
        return kern

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
                torch.empty((B, self.qkv_rows), device=dev, dtype=torch.bfloat16),
                torch.empty(
                    (B, self.num_heads, self.num_split, self.head_dim),
                    device=dev,
                    dtype=torch.float32,
                ),
                torch.empty((B, self.num_heads, self.num_split), device=dev, dtype=torch.float32),
                torch.empty((B, self.q_rows), device=dev, dtype=torch.bfloat16),
                torch.empty((B, H), device=dev, dtype=torch.float32),
                torch.empty((B, 2 * I), device=dev, dtype=torch.float32),
                self._build_queue().to(dev),
                torch.zeros(7, 40, device=dev, dtype=torch.int32),
            )
        qkv, opart, lse, attn, x2, gu, queue, bar = self._ws
        bar.zero_()  # deterministic scoreboard for this launch
        return self._kernel(
            hidden, K_cache, V_cache, Wqkv, Wo, Wgu, Wd, qkv, opart, lse, attn,
            x2, gu, queue, bar,
        )
