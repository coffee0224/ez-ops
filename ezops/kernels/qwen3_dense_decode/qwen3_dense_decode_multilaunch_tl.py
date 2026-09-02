"""Multi-launch TileLang backends for the qwen3_dense_decode op.

Stage-split variant of the persistent single-launch backend
(qwen3_dense_decode_persistent_tl): the six phases are split at the software
grid-barrier boundaries into six back-to-back kernel launches inside one
tilelang module — same flat-work-list grids, same per-phase code, barriers
replaced by kernel boundaries (a completed launch is a stronger sync than
the atomic sense-reversal barrier, so the wrapper no longer needs the
[Counter, Sense] state or its memset).

Two registered variants share this file:

  - ``tilelang_multilaunch``      plain sequential launches (ablation control)
  - ``tilelang_multilaunch_pdl``  every launch carries Programmatic Dependent
                                  Launch: T.pdl_sync() at the top of each
                                  kernel (griddepcontrol.wait — blocks until
                                  the previous kernel's writes are visible)
                                  and T.pdl_trigger() right after it
                                  (griddepcontrol.launch_dependents — releases
                                  the next kernel's staging as soon as every
                                  block has started, the aggressive-but-safe
                                  position since memory visibility is always
                                  enforced by the consumer's pdl_sync). The
                                  variant compiles with the nvrtc execution
                                  backend — the only one whose host launcher
                                  sets the PROGRAMMATIC_STREAM_SERIALIZATION
                                  launch attribute that actually activates
                                  PDL.

Because every grid is persistent-sized (num_sms CTAs) and fills the GPU,
the successor's blocks cannot actually execute before the predecessor's
CTAs retire; the expected PDL gain is the inter-kernel gap (launch
serialization + ramp) rather than compute overlap. The first kernel's
pdl_sync is a no-op (no predecessor in the module) but keeps its launch
attribute so the graph node participates in staging with whatever precedes
it in the stream.

Measured (RTX 5060 Ti, b=1, median of paired trials): under CUDA graphs
persistent / multilaunch / multilaunch_pdl are within ~1% of each other at
seq 1k-16k — the graph already pipelines node boundaries down to ~1us, and
the persistent version's five software barriers (~3us each) cost about as
much as the graph's inter-node gaps, so neither strategy wins. In EAGER
mode the split pays ~10us of GPU-side launch gap per boundary (~50us total)
and PDL recovers essentially all of it, matching the single-launch
persistent latency. PDL's device intrinsics and the
PROGRAMMATIC_STREAM_SERIALIZATION launch attribute are verified present in
the generated launcher (6 launches, 6 attrs); without the attribute the
sync/trigger pair is inert.
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


class _Qwen3DenseDecodeMultilaunchBase(BaseKernel):
    _block_n = 64  # output rows per GEMV work unit (phases 4-6)
    _block_k = 64  # K-tile of every GEMV pipeline
    _block_s = 64  # KV rows per attention tile
    _block_h = 64  # Q rows per attention gemm (tilelang layout inference)
    _num_stages = 2
    _threads = 128
    USE_PDL = False

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
        self.num_split = max(1, min(8, seq_len // 256))
        self.num_sms = get_num_sms()
        self._kernel = None
        self._ws = None  # (qkv, opart, lse, attn, x2, gu), allocated lazily once

    def _make_kernel(self):
        B, H, I = self.batch, self.hidden_size, self.intermediate_size
        NH, NKV, D = self.num_heads, self.num_kv_heads, self.head_dim
        S = self.seq_len
        QR, QKVR = self.q_rows, self.qkv_rows
        BN, BK, BS, BH = self._block_n, self._block_k, self._block_s, self._block_h
        NSTAGES, THREADS = self._num_stages, self._threads
        eps, num_sms = RMS_EPS, self.num_sms
        group, head_tiles = self.group, self.num_kv_heads
        ns = self.num_split
        use_pdl = self.USE_PDL
        # attention scale folded with log2(e): exp2(x * scale) == exp(x * sm_scale)
        scale = self.sm_scale * 1.44269504
        # per-split aligned chunk; the trailing positions ride the last split
        # through the same pipelined loop (full tiles + one [S-BLOCK_S, S)
        # window with the re-read overlap masked to -inf), see the persistent
        # backend for the layout-inference constraint this respects.
        base = (S // ns // BS) * BS
        nb = base // BS
        rem = S - ns * base
        tail_full = rem // BS
        rem_last = rem % BS
        tail_extra = 1 if rem_last > 0 else 0
        dtype, accum_dtype = "bfloat16", "float32"

        # The default tvm_ffi execution backend ignores PDL entirely: the
        # device-side griddepcontrol intrinsics are emitted, but the host
        # launch never sets CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
        # so the kernels serialize normally (verified: the repo's pdl_gemm
        # "tl_pdl" backend is inert for the same reason). The nvrtc backend
        # generates a Python launcher via cuLaunchKernelEx that adds the
        # attribute for every kernel containing T.pdl_sync().
        jit_kwargs = dict(
            out_idx=[7],
            pass_configs={PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
        )
        if use_pdl:
            jit_kwargs["execution_backend"] = "nvrtc"
        jit_dec = tilelang.jit(**jit_kwargs)

        @jit_dec
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
            ):
                # -- kernel 1: qkv = rmsnorm(hidden) @ Wqkv.T, head norms,
                #    k/v write-back; one unit = one head's D rows -----------
                with T.Kernel(num_sms, threads=THREADS) as bx:
                    Ws1 = T.alloc_shared((D, BK), dtype)
                    prod1 = T.alloc_fragment((D, BK), "float32")
                    part1 = T.alloc_fragment((D,), "float32")
                    acc1 = T.alloc_fragment((D,), "float32")
                    sq1 = T.alloc_fragment((H,), "float32")
                    sum1 = T.alloc_fragment((1,), "float32")
                    sqh = T.alloc_fragment((D,), "float32")
                    sumh = T.alloc_fragment((1,), "float32")
                    if use_pdl:
                        T.pdl_sync()
                        T.pdl_trigger()
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
                            # slot; kernel 2 then streams one uniform cache.
                            if head >= NH:
                                if head < NH + NKV:
                                    for i in T.Parallel(D):
                                        Kc[b, head - NH, S - 1, i] = acc1[i].astype(dtype)
                            if head >= NH + NKV:
                                for i in T.Parallel(D):
                                    Vc[b, head - NH - NKV, S - 1, i] = acc1[i].astype(dtype)

                # -- kernel 2: split flash-decoding attention partials -------
                with T.Kernel(num_sms, threads=THREADS) as bx:
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
                    if use_pdl:
                        T.pdl_sync()
                        T.pdl_trigger()
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
                                # remainder (full tiles + [S-BLOCK_S, S)
                                # window, overlap masked). Other splits run
                                # the extra tiles as -inf-masked dummies.
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

                # -- kernel 3: combine split partials -> Attn ---------------
                with T.Kernel(num_sms, threads=THREADS) as bx:
                    tx = T.get_thread_binding(0)
                    lse_max = T.alloc_local((1,), accum_dtype)
                    lse_log = T.alloc_local((1,), accum_dtype)
                    o_accum = T.alloc_local((1,), accum_dtype)
                    if use_pdl:
                        T.pdl_sync()
                        T.pdl_trigger()
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

                # -- kernel 4: x2 = hidden + attn @ Wo.T ---------------------
                with T.Kernel(num_sms, threads=THREADS) as bx:
                    Ws4 = T.alloc_shared((BN, BK), dtype)
                    prod4 = T.alloc_fragment((BN, BK), "float32")
                    part4 = T.alloc_fragment((BN,), "float32")
                    acc4 = T.alloc_fragment((BN,), "float32")
                    if use_pdl:
                        T.pdl_sync()
                        T.pdl_trigger()
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

                # -- kernel 5: gu = rmsnorm(x2) @ Wgu.T ----------------------
                with T.Kernel(num_sms, threads=THREADS) as bx:
                    Xs5 = T.alloc_shared((H,), dtype)
                    Ws5 = T.alloc_shared((BN, BK), dtype)
                    sq5 = T.alloc_fragment((H,), "float32")
                    sum5 = T.alloc_fragment((1,), "float32")
                    prod5 = T.alloc_fragment((BN, BK), "float32")
                    part5 = T.alloc_fragment((BN,), "float32")
                    acc5 = T.alloc_fragment((BN,), "float32")
                    if use_pdl:
                        T.pdl_sync()
                        T.pdl_trigger()
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

                # -- kernel 6: out = x2 + (silu(g)*u) @ Wd.T ------------------
                with T.Kernel(num_sms, threads=THREADS) as bx:
                    Ws6 = T.alloc_shared((BN, BK), dtype)
                    prod6 = T.alloc_fragment((BN, BK), "float32")
                    part6 = T.alloc_fragment((BN,), "float32")
                    acc6 = T.alloc_fragment((BN,), "float32")
                    if use_pdl:
                        T.pdl_sync()
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
            if self.USE_PDL:
                # tilelang's nvrtc adapter gates stream resolution on
                # str(target).startswith("cuda"), but the nvrtc target's
                # str() is the full Target JSON — so it falls back to
                # stream=0 (legacy default) and CUDA-graph capture records
                # an empty graph. Route the ambient torch stream explicitly.
                orig_torch_function = self._kernel.torch_function

                def _torch_function_with_stream(*args, _orig=orig_torch_function):
                    return _orig(*args, stream=torch.cuda.current_stream().cuda_stream)

                self._kernel.torch_function = _torch_function_with_stream
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
            )
        qkv, opart, lse, attn, x2, gu = self._ws
        return self._kernel(hidden, K_cache, V_cache, Wqkv, Wo, Wgu, Wd, qkv, opart, lse, attn, x2, gu)


@register_kernel("qwen3_dense_decode", "tilelang_multilaunch")
class Qwen3DenseDecodeMultilaunchTileLangKernel(_Qwen3DenseDecodeMultilaunchBase):
    """Six sequential launches, no PDL (ablation control for the PDL variant)."""

    USE_PDL = False


@register_kernel("qwen3_dense_decode", "tilelang_multilaunch_pdl")
class Qwen3DenseDecodeMultilaunchPdlTileLangKernel(_Qwen3DenseDecodeMultilaunchBase):
    """Six launches with Programmatic Dependent Launch on every boundary."""

    USE_PDL = True
