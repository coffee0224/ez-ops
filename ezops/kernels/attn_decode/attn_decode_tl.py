import os

os.environ.setdefault("TILELANG_CACHE_DIR", os.path.join(os.getcwd(), ".tilelang"))

import logging

import tilelang
import torch
from tilelang import language as T
from tilelang.autotuner import autotune
from tilelang.carver.arch.driver.cuda_driver import get_num_sms

logging.getLogger("tilelang.cache.kernel_cache").setLevel(logging.ERROR)

from ..base_kernel import BaseKernel
from ...registry import register_kernel


# ---------------------------------------------------------------------------
# GQA flash decoding, migrated from
# tilelang/examples/flash_decoding/example_gqa_decode.py
# ---------------------------------------------------------------------------


def _make_gqa_flash_decode_kernel(
    batch, heads, kv_heads, seq_len, dim, num_split_choices, persistent=False
):
    """JIT factory for the GQA decode kernel, migrated from tilelang's
    examples/flash_decoding/example_gqa_decode.py.

    With persistent=True the split variant launches a fixed grid of num_sms
    CTAs instead of one CTA per work unit: each CTA walks (task, kv split)
    units via `for it in T.serial(num_iters)`, so total_units never suffers
    wave quantization and the tail-heavy last splits interleave evenly.
    Per-unit compute, aligned-chunk + tail addressing, and the combine
    kernel are identical to the non-persistent variant.

    Differences from the upstream example, needed to fit ezops' op interface:

    - Tensor layout: ezops passes K/V as [batch, kv_heads, seq_len, dim];
      the example used [batch, seq_len, groups, dim]. Indexing is adapted,
      the algorithm is unchanged.
    - No mask argument (the ezops op has none).
    - Arbitrary seq_len (odd included): the example required
      seq_len % num_split == 0 and dropped any remainder. Here every
      split's pipelined phase covers an exactly block_N-aligned chunk
      (base = floor(seq_len / num_split / block_N) * block_N, same recipe
      as the TMA backend), and the trailing seq_len - num_split * base
      positions are handled by the last split in a short tail phase with
      clamped/zero-filled loads. Aligned shapes take the identical code
      path (the tail is elided at trace time when the remainder is 0).
    - The example over-read Q (a block_H=64 tile per valid_block_H rows,
      running past the end of Q on the last block). Here the Q tile is
      loaded with a clamped row index and zero-padded, so invalid rows
      produce finite garbage that the epilogue discards.
    - Autotune configs are filtered by an estimated shared-memory budget:
      the example's sm89+ heuristic (block_N=128, num_stages=2) needs
      147 KiB of shared memory and fails to launch on sm_120 (99 KiB cap).

    Returns a tilelang jit function over (num_split, block_N, num_stages)
    that dispatches to the split variant (partial outputs + LSE in global
    workspaces, plus a combine kernel) when num_split > 1, or to the
    direct no-split variant otherwise.
    """
    group = heads // kv_heads  # Q heads sharing one KV head
    if heads % kv_heads != 0:
        raise ValueError(f"num_heads ({heads}) must be divisible by num_kv_heads ({kv_heads})")
    # Must stay 64, as in the example: the acc_s -> acc_s_cast fragment copy
    # only passes tilelang's layout inference for M=64 (M=16/32 hit a layout
    # conflict between the QK^T gemm output and the PV gemm A-operand).
    block_h = 64
    valid_block_h = min(block_h, group)
    assert heads % valid_block_h == 0

    dtype = "bfloat16"
    accum_dtype = "float"
    threads = 128
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e), matches the example

    smem_optin = getattr(
        torch.cuda.get_device_properties(torch.cuda.current_device()),
        "shared_memory_per_block_optin",
        64 * 1024,
    )
    smem_budget = smem_optin - 8 * 1024  # leave room for reduce scratch
    num_sms = get_num_sms()

    def get_configs():
        configs = []
        for num_split in num_split_choices:
            for block_n in (128, 64, 32):
                # Each split's pipelined phase must cover at least one
                # full tile; the remainder goes to the last split's tail.
                if seq_len // num_split < block_n:
                    continue
                for num_stages in (3, 2, 1):
                    smem = (
                        num_stages * 2 * block_n * dim * 2  # pipelined K/V tiles
                        + block_h * dim * 2  # Q tile
                        + valid_block_h * dim * 2  # O staging
                    )
                    if smem > smem_budget:
                        continue
                    configs.append(
                        {
                            "num_split": num_split,
                            "block_N": block_n,
                            "num_stages": num_stages,
                        }
                    )
        if not configs:
            raise ValueError(
                f"No valid config for seq_len={seq_len}, num_split in {num_split_choices}, "
                f"smem budget {smem_budget} B (need seq_len // num_split >= 32)"
            )
        return configs

    def get_pass_configs():
        # Matches the example: the T.Pipelined gemm combo trips the
        # warp-specialization pass.
        return {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}

    default_num_split = num_split_choices[0]

    @autotune(configs=get_configs(), warmup=5, rep=50)
    @tilelang.jit(out_idx=[3], target="auto", pass_configs=get_pass_configs())
    def kernel(num_split=None, block_N=None, num_stages=None):
        # NOTE: keep the closure serializable (int/float/str only) — the
        # autotuner inspects and rejects non-scalar cells.
        if num_split is None:
            num_split = default_num_split
        if block_N is None:
            block_N = 64
        if num_stages is None:
            num_stages = 2

        # KV chunking (recipe from the TMA backend, plus a tail phase):
        # splits 0..num_split-1 each run `nb` pipelined block_N tiles over
        # [sid*base, (sid+1)*base) — base is block_N-aligned so no tile
        # crosses a chunk boundary and no load ever leaves [0, seq_len).
        # The trailing `rem` positions are handled by the LAST split only,
        # in a non-pipelined tail with clamped/zero-filled loads. rem == 0
        # (aligned seq_len) elides the tail at trace time.
        base = (seq_len // num_split // block_N) * block_N
        nb = base // block_N
        rem = seq_len - num_split * base
        tail_blocks = (rem + block_N - 1) // block_N

        if persistent and num_split > 1:

            @T.prim_func
            def flashattn_gqa_decode_split_persistent(
                Q: T.Buffer((batch, heads, dim), dtype),
                K: T.Buffer((batch, kv_heads, seq_len, dim), dtype),
                V: T.Buffer((batch, kv_heads, seq_len, dim), dtype),
                Output: T.Buffer((batch, heads, dim), dtype),
            ):
                # fp32 workspaces: bf16 alloc_global trips tilelang's
                # storage-legalizer ("Cannot find var remap"), and the
                # buffer is tiny so the wider dtype is free.
                glse = T.alloc_global((batch, heads, num_split), accum_dtype)
                Output_partial = T.alloc_global((batch, heads, num_split, dim), accum_dtype)

                # persistent split: exactly num_sms CTAs, each walking
                # (task, kv split) work units. Split-major decode keeps
                # the heavier last-split units spread across CTAs.
                head_tiles = heads // valid_block_h
                total_units = batch * head_tiles * num_split
                num_iters = (total_units + num_sms - 1) // num_sms

                with T.Kernel(num_sms, threads=threads) as bx:
                    Q_shared = T.alloc_shared((block_h, dim), dtype)
                    K_shared = T.alloc_shared((block_N, dim), dtype)
                    V_shared = T.alloc_shared((block_N, dim), dtype)
                    O_shared = T.alloc_shared((valid_block_h, dim), accum_dtype)
                    acc_s = T.alloc_fragment((block_h, block_N), accum_dtype)
                    acc_s_cast = T.alloc_fragment((block_h, block_N), dtype)
                    acc_o = T.alloc_fragment((block_h, dim), accum_dtype)
                    scores_max = T.alloc_fragment((block_h,), accum_dtype)
                    scores_max_prev = T.alloc_fragment((block_h,), accum_dtype)
                    scores_scale = T.alloc_fragment((block_h,), accum_dtype)
                    scores_sum = T.alloc_fragment((block_h,), accum_dtype)
                    logsum = T.alloc_fragment((block_h,), accum_dtype)

                    for it in T.serial(num_iters):
                        work_id = it * num_sms + bx
                        if work_id < total_units:
                            sid = work_id % num_split
                            task = work_id // num_split
                            bid = task // head_tiles
                            hid = task % head_tiles
                            q_row0 = hid * valid_block_h
                            cur_kv_head = q_row0 // group

                            # Load this block's valid Q rows, zero-pad the
                            # rest of the tile (clamped read: never past
                            # the end of Q).
                            for i, j in T.Parallel(block_h, dim):
                                Q_shared[i, j] = T.if_then_else(
                                    i < valid_block_h,
                                    Q[bid, T.min(q_row0 + i, heads - 1), j],
                                    T.cast(0, dtype),
                                )
                            T.fill(acc_o, 0)
                            T.fill(logsum, 0)
                            T.fill(scores_max, -T.infinity(accum_dtype))

                            for k in T.Pipelined(nb, num_stages=num_stages):
                                kv_start = sid * base + k * block_N
                                T.copy(K[bid, cur_kv_head, kv_start : kv_start + block_N, :], K_shared)
                                T.copy(V[bid, cur_kv_head, kv_start : kv_start + block_N, :], V_shared)
                                T.clear(acc_s)
                                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                T.copy(scores_max, scores_max_prev)
                                T.fill(scores_max, -T.infinity(accum_dtype))
                                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                                for i in T.Parallel(block_h):
                                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                                for i in T.Parallel(block_h):
                                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                                for i, j in T.Parallel(block_h, block_N):
                                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                                T.reduce_sum(acc_s, scores_sum, dim=1)
                                for i in T.Parallel(block_h):
                                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                                T.copy(acc_s, acc_s_cast)
                                for i, j in T.Parallel(block_h, dim):
                                    acc_o[i, j] *= scores_scale[i]
                                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                            # Remainder tail (odd seq_len etc.): the last
                            # split picks up the positions the aligned
                            # chunks could not cover; guarded loads keep
                            # every read inside [0, seq_len).
                            if sid == num_split - 1:
                                for tb in T.serial(tail_blocks):
                                    t0 = num_split * base + tb * block_N
                                    for i, j in T.Parallel(block_N, dim):
                                        pos = t0 + i
                                        K_shared[i, j] = T.if_then_else(
                                            pos < seq_len,
                                            K[bid, cur_kv_head, T.min(pos, seq_len - 1), j],
                                            T.cast(0, dtype),
                                        )
                                    for i, j in T.Parallel(block_N, dim):
                                        pos = t0 + i
                                        V_shared[i, j] = T.if_then_else(
                                            pos < seq_len,
                                            V[bid, cur_kv_head, T.min(pos, seq_len - 1), j],
                                            T.cast(0, dtype),
                                        )
                                    T.clear(acc_s)
                                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                    for i, j in T.Parallel(block_h, block_N):
                                        acc_s[i, j] = T.if_then_else(
                                            t0 + j < seq_len,
                                            acc_s[i, j],
                                            -T.infinity(accum_dtype),
                                        )
                                    T.copy(scores_max, scores_max_prev)
                                    T.fill(scores_max, -T.infinity(accum_dtype))
                                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                                    for i in T.Parallel(block_h):
                                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                                    for i in T.Parallel(block_h):
                                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                                    for i, j in T.Parallel(block_h, block_N):
                                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                                    T.reduce_sum(acc_s, scores_sum, dim=1)
                                    for i in T.Parallel(block_h):
                                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                                    T.copy(acc_s, acc_s_cast)
                                    for i, j in T.Parallel(block_h, dim):
                                        acc_o[i, j] *= scores_scale[i]
                                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                            for i, j in T.Parallel(block_h, dim):
                                acc_o[i, j] /= logsum[i]
                            for i in T.Parallel(block_h):
                                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                            for i in T.Parallel(block_h):
                                if i < valid_block_h:
                                    glse[bid, q_row0 + i, sid] = logsum[i]
                            T.copy(acc_o[:valid_block_h, :], O_shared)
                            T.copy(O_shared, Output_partial[bid, q_row0 : q_row0 + valid_block_h, sid, :])

                # combine: weighted merge of the per-split partial outputs
                with T.Kernel(batch * heads, threads=threads) as bx:
                    lane_id = T.get_thread_binding(0)
                    brow = bx // heads
                    hcol = bx % heads

                    lse_max = T.alloc_local((1,), accum_dtype)
                    lse_max[0] = -T.infinity(accum_dtype)
                    for s in T.serial(num_split):
                        lse_max[0] = T.max(lse_max[0], glse[brow, hcol, s])
                    lse_log = T.alloc_local((1,), accum_dtype)
                    lse_log[0] = 0.0
                    for s in T.serial(num_split):
                        lse_log[0] += T.exp2(glse[brow, hcol, s] - lse_max[0])
                    lse_log[0] = T.log2(lse_log[0]) + lse_max[0]

                    o_accum = T.alloc_local((1,), accum_dtype)
                    for i in T.serial(T.ceildiv(dim, threads)):
                        idx = i * threads + lane_id
                        if idx < dim:
                            o_accum[0] = 0.0
                            for s in T.serial(num_split):
                                w = T.exp2(glse[brow, hcol, s] - lse_log[0])
                                o_accum[0] += Output_partial[brow, hcol, s, idx].astype(accum_dtype) * w
                            Output[brow, hcol, idx] = o_accum[0].astype(dtype)

            return flashattn_gqa_decode_split_persistent

        if num_split > 1:

            @T.prim_func
            def flashattn_gqa_decode_split(
                Q: T.Buffer((batch, heads, dim), dtype),
                K: T.Buffer((batch, kv_heads, seq_len, dim), dtype),
                V: T.Buffer((batch, kv_heads, seq_len, dim), dtype),
                Output: T.Buffer((batch, heads, dim), dtype),
            ):
                # fp32 workspaces: bf16 alloc_global trips tilelang's
                # storage-legalizer ("Cannot find var remap"), and the
                # buffer is tiny so the wider dtype is free.
                glse = T.alloc_global((batch, heads, num_split), accum_dtype)
                Output_partial = T.alloc_global((batch, heads, num_split, dim), accum_dtype)

                # split: one CTA per (batch, valid_block_h Q rows, kv split)
                with T.Kernel(batch, heads // valid_block_h, num_split, threads=threads) as (bx, by, bz):
                    Q_shared = T.alloc_shared((block_h, dim), dtype)
                    K_shared = T.alloc_shared((block_N, dim), dtype)
                    V_shared = T.alloc_shared((block_N, dim), dtype)
                    O_shared = T.alloc_shared((valid_block_h, dim), accum_dtype)
                    acc_s = T.alloc_fragment((block_h, block_N), accum_dtype)
                    acc_s_cast = T.alloc_fragment((block_h, block_N), dtype)
                    acc_o = T.alloc_fragment((block_h, dim), accum_dtype)
                    scores_max = T.alloc_fragment((block_h,), accum_dtype)
                    scores_max_prev = T.alloc_fragment((block_h,), accum_dtype)
                    scores_scale = T.alloc_fragment((block_h,), accum_dtype)
                    scores_sum = T.alloc_fragment((block_h,), accum_dtype)
                    logsum = T.alloc_fragment((block_h,), accum_dtype)

                    bid = bx
                    hid = by
                    sid = bz
                    q_row0 = hid * valid_block_h
                    cur_kv_head = q_row0 // group

                    # Load this block's valid Q rows, zero-pad the rest of
                    # the tile (clamped read: never past the end of Q).
                    for i, j in T.Parallel(block_h, dim):
                        Q_shared[i, j] = T.if_then_else(
                            i < valid_block_h,
                            Q[bid, T.min(q_row0 + i, heads - 1), j],
                            T.cast(0, dtype),
                        )
                    T.fill(acc_o, 0)
                    T.fill(logsum, 0)
                    T.fill(scores_max, -T.infinity(accum_dtype))

                    for k in T.Pipelined(nb, num_stages=num_stages):
                        kv_start = sid * base + k * block_N
                        T.copy(K[bid, cur_kv_head, kv_start : kv_start + block_N, :], K_shared)
                        T.copy(V[bid, cur_kv_head, kv_start : kv_start + block_N, :], V_shared)
                        T.clear(acc_s)
                        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(accum_dtype))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_h):
                            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                        for i in T.Parallel(block_h):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                        for i, j in T.Parallel(block_h, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_h):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        T.copy(acc_s, acc_s_cast)
                        for i, j in T.Parallel(block_h, dim):
                            acc_o[i, j] *= scores_scale[i]
                        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                    # Remainder tail (odd seq_len etc.): the last split picks
                    # up the positions the aligned chunks could not cover.
                    # Non-pipelined loads with clamped index + zero fill keep
                    # every read inside [0, seq_len); masked scores get -inf.
                    if sid == num_split - 1:
                        for tb in T.serial(tail_blocks):
                            t0 = num_split * base + tb * block_N
                            for i, j in T.Parallel(block_N, dim):
                                pos = t0 + i
                                K_shared[i, j] = T.if_then_else(
                                    pos < seq_len,
                                    K[bid, cur_kv_head, T.min(pos, seq_len - 1), j],
                                    T.cast(0, dtype),
                                )
                            for i, j in T.Parallel(block_N, dim):
                                pos = t0 + i
                                V_shared[i, j] = T.if_then_else(
                                    pos < seq_len,
                                    V[bid, cur_kv_head, T.min(pos, seq_len - 1), j],
                                    T.cast(0, dtype),
                                )
                            T.clear(acc_s)
                            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                            for i, j in T.Parallel(block_h, block_N):
                                acc_s[i, j] = T.if_then_else(
                                    t0 + j < seq_len,
                                    acc_s[i, j],
                                    -T.infinity(accum_dtype),
                                )
                            T.copy(scores_max, scores_max_prev)
                            T.fill(scores_max, -T.infinity(accum_dtype))
                            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                            for i in T.Parallel(block_h):
                                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                            for i in T.Parallel(block_h):
                                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                            for i, j in T.Parallel(block_h, block_N):
                                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                            T.reduce_sum(acc_s, scores_sum, dim=1)
                            for i in T.Parallel(block_h):
                                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                            T.copy(acc_s, acc_s_cast)
                            for i, j in T.Parallel(block_h, dim):
                                acc_o[i, j] *= scores_scale[i]
                            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                    for i, j in T.Parallel(block_h, dim):
                        acc_o[i, j] /= logsum[i]
                    for i in T.Parallel(block_h):
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                    for i in T.Parallel(block_h):
                        if i < valid_block_h:
                            glse[bid, q_row0 + i, sid] = logsum[i]
                    T.copy(acc_o[:valid_block_h, :], O_shared)
                    T.copy(O_shared, Output_partial[bid, q_row0 : q_row0 + valid_block_h, sid, :])

                # combine: weighted merge of the per-split partial outputs
                with T.Kernel(batch * heads, threads=threads) as bx:
                    lane_id = T.get_thread_binding(0)
                    brow = bx // heads
                    hcol = bx % heads

                    lse_max = T.alloc_local((1,), accum_dtype)
                    lse_max[0] = -T.infinity(accum_dtype)
                    for s in T.serial(num_split):
                        lse_max[0] = T.max(lse_max[0], glse[brow, hcol, s])
                    lse_log = T.alloc_local((1,), accum_dtype)
                    lse_log[0] = 0.0
                    for s in T.serial(num_split):
                        lse_log[0] += T.exp2(glse[brow, hcol, s] - lse_max[0])
                    lse_log[0] = T.log2(lse_log[0]) + lse_max[0]

                    o_accum = T.alloc_local((1,), accum_dtype)
                    for i in T.serial(T.ceildiv(dim, threads)):
                        idx = i * threads + lane_id
                        if idx < dim:
                            o_accum[0] = 0.0
                            for s in T.serial(num_split):
                                w = T.exp2(glse[brow, hcol, s] - lse_log[0])
                                o_accum[0] += Output_partial[brow, hcol, s, idx].astype(accum_dtype) * w
                            Output[brow, hcol, idx] = o_accum[0].astype(dtype)

            return flashattn_gqa_decode_split

        @T.prim_func
        def flashattn_gqa_decode_no_split(
            Q: T.Buffer((batch, heads, dim), dtype),
            K: T.Buffer((batch, kv_heads, seq_len, dim), dtype),
            V: T.Buffer((batch, kv_heads, seq_len, dim), dtype),
            Output: T.Buffer((batch, heads, dim), dtype),
        ):
            with T.Kernel(batch, heads // valid_block_h, 1, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared((block_h, dim), dtype)
                K_shared = T.alloc_shared((block_N, dim), dtype)
                V_shared = T.alloc_shared((block_N, dim), dtype)
                O_shared = T.alloc_shared((valid_block_h, dim), dtype)
                acc_s = T.alloc_fragment((block_h, block_N), accum_dtype)
                acc_s_cast = T.alloc_fragment((block_h, block_N), dtype)
                acc_o = T.alloc_fragment((block_h, dim), accum_dtype)
                scores_max = T.alloc_fragment((block_h,), accum_dtype)
                scores_max_prev = T.alloc_fragment((block_h,), accum_dtype)
                scores_scale = T.alloc_fragment((block_h,), accum_dtype)
                scores_sum = T.alloc_fragment((block_h,), accum_dtype)
                logsum = T.alloc_fragment((block_h,), accum_dtype)

                bid = bx
                hid = by
                q_row0 = hid * valid_block_h
                cur_kv_head = q_row0 // group

                for i, j in T.Parallel(block_h, dim):
                    Q_shared[i, j] = T.if_then_else(
                        i < valid_block_h,
                        Q[bid, T.min(q_row0 + i, heads - 1), j],
                        T.cast(0, dtype),
                    )
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                for k in T.Pipelined(nb, num_stages=num_stages):
                    kv_start = k * block_N
                    T.copy(K[bid, cur_kv_head, kv_start : kv_start + block_N, :], K_shared)
                    T.copy(V[bid, cur_kv_head, kv_start : kv_start + block_N, :], V_shared)
                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_h):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(block_h):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_h, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_h):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(block_h, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                # Remainder tail (odd seq_len etc.), same guarded scheme as
                # the split kernel's tail; traced only when rem > 0.
                if tail_blocks > 0:
                    for tb in T.serial(tail_blocks):
                        t0 = base + tb * block_N
                        for i, j in T.Parallel(block_N, dim):
                            pos = t0 + i
                            K_shared[i, j] = T.if_then_else(
                                pos < seq_len,
                                K[bid, cur_kv_head, T.min(pos, seq_len - 1), j],
                                T.cast(0, dtype),
                            )
                        for i, j in T.Parallel(block_N, dim):
                            pos = t0 + i
                            V_shared[i, j] = T.if_then_else(
                                pos < seq_len,
                                V[bid, cur_kv_head, T.min(pos, seq_len - 1), j],
                                T.cast(0, dtype),
                            )
                        T.clear(acc_s)
                        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        for i, j in T.Parallel(block_h, block_N):
                            acc_s[i, j] = T.if_then_else(
                                t0 + j < seq_len,
                                acc_s[i, j],
                                -T.infinity(accum_dtype),
                            )
                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(accum_dtype))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_h):
                            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                        for i in T.Parallel(block_h):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                        for i, j in T.Parallel(block_h, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_h):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        T.copy(acc_s, acc_s_cast)
                        for i, j in T.Parallel(block_h, dim):
                            acc_o[i, j] *= scores_scale[i]
                        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_h, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o[:valid_block_h, :], O_shared)
                T.copy(O_shared, Output[bid, q_row0 : q_row0 + valid_block_h, :])

        return flashattn_gqa_decode_no_split

    return kernel


@register_kernel("attn_decode", "flash_decode_tilelang_gqa")
class GqaFlashDecodeTileLangKernel(BaseKernel):
    """GQA decode without KV splitting (num_split=1).

    Migrated from tilelang's example_gqa_decode.py: each CTA runs online-
    softmax flash attention over the full KV length for valid_block_h Q
    rows of one KV group, tensor-core gemms for QK^T and PV.
    """

    _num_split_choices = (1,)
    _persistent = False
    supports_gqa = True

    def __init__(self, batch, num_heads, seq_len, head_dim, num_kv_heads=None):
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.batch = batch
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self._factory = _make_gqa_flash_decode_kernel(
            batch,
            num_heads,
            num_kv_heads,
            seq_len,
            head_dim,
            self._num_split_choices,
            persistent=self._persistent,
        )
        self._kernel = None

    def __call__(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, H, _, D = Q.shape
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
        assert (B, H, D) == (self.batch, self.num_heads, self.head_dim)
        assert (K.shape[1], K.shape[2], D) == (self.num_kv_heads, self.seq_len, self.head_dim)
        assert Q.dtype == torch.bfloat16

        if self._kernel is None:
            self._kernel = self._factory()
        out = self._kernel(Q.reshape(B, H, D), K, V)
        return out.reshape(B, H, 1, D)


@register_kernel("attn_decode", "flash_decode_tilelang_gqa_split")
class GqaFlashDecodeSplitTileLangKernel(GqaFlashDecodeTileLangKernel):
    """GQA decode with KV splitting (num_split autotuned over divisors).

    Same kernel as the no-split variant, but each CTA handles a
    seq_len/num_split KV chunk and writes (partial output, LSE) to global
    workspaces; a small combine kernel merges the partials with
    log-sum-exp weights. Splits the KV loop across CTAs to fill SMs when
    batch * Q head tiles are too few.
    """

    _num_split_choices = (2, 4, 8)


@register_kernel("attn_decode", "flash_decode_tilelang_gqa_split_persistent")
class GqaFlashDecodeSplitPersistentTileLangKernel(GqaFlashDecodeSplitTileLangKernel):
    """Persistent-grid variant of flash_decode_tilelang_gqa_split.

    Identical compute, addressing, and combine pass, but the grid is fixed
    to num_sms CTAs that pull (task, kv split) work units from a flat list
    instead of one CTA per unit. Removes wave quantization (e.g. 128 units
    on 36 SMs = 3.55 ragged waves) at the cost of a pipeline drain between
    consecutive units on the same CTA; num_split still autotunes the chunk
    granularity.
    """

    _num_split_choices = (2, 4, 8)
    _persistent = True
