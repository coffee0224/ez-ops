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


class _FlashDecodeTileLangBase(BaseKernel):
    """Shared scaffolding for the flash-decoding tilelang backends.

    Persistent grid (num_sms) iterates over (task, kv_split) work units so all
    SMs stay busy. Each work unit processes its KV chunk with per-warp online
    softmax, then a cross-warp reduction writes (partial_output, LSE) to
    workspaces. A second tiny kernel combines the per-split outputs.

    Subclasses override `_make_split_kernel` to choose the KV-load strategy
    (TMA tile loads vs direct LDG).
    """

    def __init__(self, batch, num_heads, seq_len, head_dim):
        self.batch = batch
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.num_sms = get_num_sms()
        self.num_split = self._pick_num_split()
        self._split_kernel = self._make_split_kernel()
        self._combine_kernel = self._make_combine_kernel()
        self._best_split = None
        self._combine_built = None

    def _pick_num_split(self):
        """Pick num_split so batch*num_heads*num_split fills all SMs evenly.

        For (1,16) on 36 SMs: smallest k where 16*k is a multiple of 36 is k=9
        (144 units, 4/SM). Also require seq_len % k == 0 for clean chunking.
        """
        total_tasks = self.batch * self.num_heads
        if total_tasks <= 0:
            return 1
        min_split = (self.num_sms + total_tasks - 1) // total_tasks
        for cand in range(min_split, min_split + self.num_sms):
            if (
                self.seq_len % cand == 0
                and (total_tasks * cand) % self.num_sms == 0
            ):
                return cand
        for cand in range(min_split, min_split + self.num_sms):
            if self.seq_len % cand == 0:
                return cand
        return min_split

    def _make_split_kernel(self):
        raise NotImplementedError

    def _make_combine_kernel(self):
        total_tasks = self.batch * self.num_heads
        num_split = self.num_split
        head_dim = self.head_dim

        @tilelang.jit(out_idx=[2], target="auto")
        def kernel():
            accum_dtype = "float"
            dtype = "bfloat16"

            @T.prim_func
            def main(
                Out_partial: T.Buffer((total_tasks, num_split, head_dim), dtype),
                LSE_partial: T.Buffer((total_tasks, num_split), accum_dtype),
                Out: T.Buffer((total_tasks, head_dim), dtype),
            ):
                with T.Kernel(total_tasks, threads=32) as task_id:
                    lane_id = T.get_thread_binding(0)

                    cur_max = T.alloc_local((1,), accum_dtype)
                    cur_max[0] = -1e30
                    for s in T.serial(num_split):
                        cur_max[0] = T.max(
                            cur_max[0], LSE_partial[task_id, s]
                        )

                    sum_w = T.alloc_local((1,), accum_dtype)
                    sum_w[0] = 0.0
                    oo0 = T.alloc_local((1,), accum_dtype)
                    oo1 = T.alloc_local((1,), accum_dtype)
                    oo2 = T.alloc_local((1,), accum_dtype)
                    oo3 = T.alloc_local((1,), accum_dtype)
                    oo0[0] = 0.0
                    oo1[0] = 0.0
                    oo2[0] = 0.0
                    oo3[0] = 0.0

                    for s in T.serial(num_split):
                        w = T.exp2(LSE_partial[task_id, s] - cur_max[0])
                        oo0[0] += w * Out_partial[
                            task_id, s, 0 * 32 + lane_id
                        ].astype(accum_dtype)
                        oo1[0] += w * Out_partial[
                            task_id, s, 1 * 32 + lane_id
                        ].astype(accum_dtype)
                        oo2[0] += w * Out_partial[
                            task_id, s, 2 * 32 + lane_id
                        ].astype(accum_dtype)
                        oo3[0] += w * Out_partial[
                            task_id, s, 3 * 32 + lane_id
                        ].astype(accum_dtype)
                        sum_w[0] += w

                    Out[task_id, 0 * 32 + lane_id] = (oo0[0] / sum_w[0]).astype(
                        dtype
                    )
                    Out[task_id, 1 * 32 + lane_id] = (oo1[0] / sum_w[0]).astype(
                        dtype
                    )
                    Out[task_id, 2 * 32 + lane_id] = (oo2[0] / sum_w[0]).astype(
                        dtype
                    )
                    Out[task_id, 3 * 32 + lane_id] = (oo3[0] / sum_w[0]).astype(
                        dtype
                    )

            return main

        return kernel

    def __call__(self, Q, K, V):
        B, H, _, D = Q.shape
        S = K.shape[2]
        Q_flat = Q.reshape(B * H, D)
        K_flat = K.reshape(B * H, S, D)
        V_flat = V.reshape(B * H, S, D)

        device = Q.device
        dtype = torch.bfloat16
        total_tasks = B * H
        Out_partial = torch.empty(
            (total_tasks, self.num_split, D), device=device, dtype=dtype
        )
        LSE_partial = torch.empty(
            (total_tasks, self.num_split), device=device, dtype=torch.float32
        )

        if self._best_split is None:
            self._best_split = self._split_kernel()
            self._combine_built = self._combine_kernel()

        self._best_split(Q_flat, K_flat, V_flat, Out_partial, LSE_partial)
        Out_flat = self._combine_built(Out_partial, LSE_partial)
        return Out_flat.reshape(B, H, 1, D)


@register_kernel("attn_decode", "flash_decode_tilelang_tma")
class FlashDecodeAttnTileLangKernel(_FlashDecodeTileLangBase):
    """TMA-based flash decoding.

    Outer loop iterates over `block_n`-sized KV tiles loaded via `T.copy`
    (lowered to TMA on Blackwell sm_120). `T.Pipelined(num_stages=2)`
    overlaps the next tile's load with the current tile's compute. The
    inner dot-product uses `T.warp_reduce_sum` (shfl) because
    `T.tvm_thread_allreduce` cannot be hosted inside `T.Pipelined`.

    Unlike the LDG variant, this backend prefers **perfect SM balance**
    over `seq_len` divisibility: num_split is picked so
    `(total_tasks * num_split) % num_sms == 0`, and the remainder of
    `seq_len / num_split` is dumped into the last split (slightly larger
    chunk). Non-last splits may load a partial tail block whose OOB
    positions land in the next split's data — those positions get their
    softmax score masked to -inf so they contribute zero weight.
    """

    def _pick_num_split(self):
        """Perfect-SM-balance num_split (drops seq_len divisibility)."""
        total_tasks = self.batch * self.num_heads
        if total_tasks <= 0:
            return 1
        min_split = (self.num_sms + total_tasks - 1) // total_tasks
        for cand in range(min_split, min_split + self.num_sms * 4):
            if (total_tasks * cand) % self.num_sms == 0:
                return cand
        return min_split

    def _make_split_kernel(self):
        total_tasks = self.batch * self.num_heads
        num_split = self.num_split
        total_work_units = total_tasks * num_split
        num_iters = (total_work_units + self.num_sms - 1) // self.num_sms
        seq_len = self.seq_len
        head_dim = self.head_dim
        num_sms = self.num_sms

        # Unequal-chunk distribution:
        # - Splits 0..num_split-2: kv_chunk_base positions each
        # - Last split: remainder (>= kv_chunk_base), so the last split's
        #   chunk is always a multiple of any block_n that divides the
        #   remainder — keeps the final block of the final split inside
        #   the K/V buffer (no OOB past seq_len).
        kv_chunk_base = seq_len // num_split
        last_split_chunk = seq_len - (num_split - 1) * kv_chunk_base
        kv_chunk_max = max(kv_chunk_base, last_split_chunk)

        # Restrict block_n to values that keep the last split's last block
        # within seq_len (last_split_chunk % block_n == 0). Non-last splits
        # may have partial tail blocks — those reads land in the next
        # split's data within the K/V buffer, which is safe.
        all_block_ns = [32, 64, 128]
        safe_block_ns = [bn for bn in all_block_ns if last_split_chunk % bn == 0]
        if not safe_block_ns:
            safe_block_ns = all_block_ns
        default_block_n = safe_block_ns[0]

        def get_configs():
            configs = []
            for nw in [2, 4, 8]:
                for bn in safe_block_ns:
                    if bn % nw == 0:
                        configs.append({"num_warps": nw, "block_n": bn})
            return configs

        def get_pass_configs():
            # T.Pipelined + cross-warp allreduce combo trips the producer-
            # consumer warp-specialization pass; disable it (matches the
            # tilelang flash_decoding example).
            return {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}

        @autotune(configs=get_configs(), warmup=5, rep=50)
        @tilelang.jit(target="auto", pass_configs=get_pass_configs())
        def kernel(num_warps=None, block_n=None):
            dtype = "bfloat16"
            accum_dtype = "float"
            attn_scale = 1.0 / (head_dim**0.5)
            log2e = 1.44269504
            neg_inf = -1e30
            sum_red = T.comm_reducer(lambda x, y: x + y, [T.cast(0, accum_dtype)])
            max_red = T.comm_reducer(lambda x, y: T.max(x, y), [T.cast(neg_inf, accum_dtype)])

            if num_warps is None:
                num_warps = 4
            if block_n is None:
                block_n = default_block_n

            pos_per_warp = block_n // num_warps
            num_blocks = (kv_chunk_max + block_n - 1) // block_n

            @T.prim_func
            def main(
                Q: T.Buffer((total_tasks, head_dim), dtype),
                K: T.Buffer((total_tasks, seq_len, head_dim), dtype),
                V: T.Buffer((total_tasks, seq_len, head_dim), dtype),
                Out_partial: T.Buffer((total_tasks, num_split, head_dim), dtype),
                LSE_partial: T.Buffer((total_tasks, num_split), accum_dtype),
            ):
                with T.Kernel(num_sms, threads=(32, num_warps)) as block_id:
                    lane_id = T.get_thread_binding(0)
                    warp_id = T.get_thread_binding(1)

                    Q_shared = T.alloc_shared((head_dim,), accum_dtype)
                    K_shared = T.alloc_shared((block_n, head_dim), dtype)
                    V_shared = T.alloc_shared((block_n, head_dim), dtype)

                    for it in T.serial(num_iters):
                        work_id = it * num_sms + block_id
                        if work_id < total_work_units:
                            split_id = work_id % num_split
                            task_id = work_id // num_split
                            kv_start = split_id * kv_chunk_base
                            # Last split ends at seq_len; others end at the
                            # next split's start. OOB positions (in tail
                            # blocks of non-last splits) are masked below.
                            kv_end = T.if_then_else(
                                split_id == num_split - 1,
                                seq_len,
                                (split_id + 1) * kv_chunk_base,
                            )

                            for d in T.Parallel(head_dim):
                                Q_shared[d] = Q[task_id, d].astype(accum_dtype)

                            q0 = Q_shared[0 * 32 + lane_id]
                            q1 = Q_shared[1 * 32 + lane_id]
                            q2 = Q_shared[2 * 32 + lane_id]
                            q3 = Q_shared[3 * 32 + lane_id]

                            o0 = T.alloc_local((1,), accum_dtype)
                            o1 = T.alloc_local((1,), accum_dtype)
                            o2 = T.alloc_local((1,), accum_dtype)
                            o3 = T.alloc_local((1,), accum_dtype)
                            T.clear(o0)
                            T.clear(o1)
                            T.clear(o2)
                            T.clear(o3)
                            ms = T.alloc_local((1,), accum_dtype)
                            ms[0] = neg_inf
                            ls = T.alloc_local((1,), accum_dtype)
                            ls[0] = 0.0

                            # Pipelined TMA tile loads + per-position SIMT compute.
                            # Inner dot-product uses warp-level shfl reduction
                            # (T.warp_reduce_sum) instead of tvm_thread_allreduce
                            # because allreduce cannot be hosted inside T.Pipelined.
                            for kb in T.Pipelined(num_blocks, num_stages=2):
                                block_start = kv_start + kb * block_n

                                T.copy(
                                    K[
                                        task_id,
                                        block_start : block_start + block_n,
                                        :,
                                    ],
                                    K_shared,
                                )
                                T.copy(
                                    V[
                                        task_id,
                                        block_start : block_start + block_n,
                                        :,
                                    ],
                                    V_shared,
                                )

                                for j in T.serial(pos_per_warp):
                                    local_pos = warp_id * pos_per_warp + j
                                    global_pos = block_start + local_pos

                                    k0 = K_shared[
                                        local_pos, 0 * 32 + lane_id
                                    ].astype(accum_dtype)
                                    k1 = K_shared[
                                        local_pos, 1 * 32 + lane_id
                                    ].astype(accum_dtype)
                                    k2 = K_shared[
                                        local_pos, 2 * 32 + lane_id
                                    ].astype(accum_dtype)
                                    k3 = K_shared[
                                        local_pos, 3 * 32 + lane_id
                                    ].astype(accum_dtype)

                                    sp = T.alloc_local((1,), accum_dtype)
                                    sp[0] = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3

                                    sc = T.alloc_local((1,), accum_dtype)
                                    sc[0] = T.warp_reduce_sum(sp[0]) * attn_scale
                                    # Mask out OOB positions (read from next
                                    # split's data within the K/V buffer) so
                                    # they contribute zero softmax weight.
                                    sc[0] = T.if_then_else(
                                        global_pos >= kv_end, neg_inf, sc[0]
                                    )

                                    om = T.alloc_local((1,), accum_dtype)
                                    om[0] = ms[0]
                                    ms[0] = T.max(ms[0], sc[0])
                                    ed = T.exp2((om[0] - ms[0]) * log2e)
                                    wt = T.exp2((sc[0] - ms[0]) * log2e)
                                    ls[0] = ls[0] * ed + wt

                                    v0 = V_shared[
                                        local_pos, 0 * 32 + lane_id
                                    ].astype(accum_dtype)
                                    v1 = V_shared[
                                        local_pos, 1 * 32 + lane_id
                                    ].astype(accum_dtype)
                                    v2 = V_shared[
                                        local_pos, 2 * 32 + lane_id
                                    ].astype(accum_dtype)
                                    v3 = V_shared[
                                        local_pos, 3 * 32 + lane_id
                                    ].astype(accum_dtype)

                                    o0[0] = o0[0] * ed + wt * v0
                                    o1[0] = o1[0] * ed + wt * v1
                                    o2[0] = o2[0] * ed + wt * v2
                                    o3[0] = o3[0] * ed + wt * v3

                            # Cross-warp reduction (inlined; helper trips the
                            # tilelang AST builder).
                            bm = T.alloc_local((1,), accum_dtype)
                            with T.attr(
                                max_red,
                                "reduce_scope",
                                T.reinterpret(T.uint64(0), dtype="handle"),
                            ):
                                T.evaluate(
                                    T.tvm_thread_allreduce(
                                        T.uint32(1),
                                        ms[0],
                                        True,
                                        bm[0],
                                        warp_id,
                                        dtype="handle",
                                    )
                                )

                            ws = T.exp2((ms[0] - bm[0]) * log2e)

                            cs = T.alloc_local((1,), accum_dtype)
                            cs[0] = ls[0] * ws
                            bs = T.alloc_local((1,), accum_dtype)
                            with T.attr(
                                sum_red,
                                "reduce_scope",
                                T.reinterpret(T.uint64(0), dtype="handle"),
                            ):
                                T.evaluate(
                                    T.tvm_thread_allreduce(
                                        T.uint32(1),
                                        cs[0],
                                        True,
                                        bs[0],
                                        warp_id,
                                        dtype="handle",
                                    )
                                )

                            c0 = T.alloc_local((1,), accum_dtype)
                            c0[0] = o0[0] * ws
                            bo0 = T.alloc_local((1,), accum_dtype)
                            with T.attr(
                                sum_red,
                                "reduce_scope",
                                T.reinterpret(T.uint64(0), dtype="handle"),
                            ):
                                T.evaluate(
                                    T.tvm_thread_allreduce(
                                        T.uint32(1),
                                        c0[0],
                                        True,
                                        bo0[0],
                                        warp_id,
                                        dtype="handle",
                                    )
                                )

                            c1 = T.alloc_local((1,), accum_dtype)
                            c1[0] = o1[0] * ws
                            bo1 = T.alloc_local((1,), accum_dtype)
                            with T.attr(
                                sum_red,
                                "reduce_scope",
                                T.reinterpret(T.uint64(0), dtype="handle"),
                            ):
                                T.evaluate(
                                    T.tvm_thread_allreduce(
                                        T.uint32(1),
                                        c1[0],
                                        True,
                                        bo1[0],
                                        warp_id,
                                        dtype="handle",
                                    )
                                )

                            c2 = T.alloc_local((1,), accum_dtype)
                            c2[0] = o2[0] * ws
                            bo2 = T.alloc_local((1,), accum_dtype)
                            with T.attr(
                                sum_red,
                                "reduce_scope",
                                T.reinterpret(T.uint64(0), dtype="handle"),
                            ):
                                T.evaluate(
                                    T.tvm_thread_allreduce(
                                        T.uint32(1),
                                        c2[0],
                                        True,
                                        bo2[0],
                                        warp_id,
                                        dtype="handle",
                                    )
                                )

                            c3 = T.alloc_local((1,), accum_dtype)
                            c3[0] = o3[0] * ws
                            bo3 = T.alloc_local((1,), accum_dtype)
                            with T.attr(
                                sum_red,
                                "reduce_scope",
                                T.reinterpret(T.uint64(0), dtype="handle"),
                            ):
                                T.evaluate(
                                    T.tvm_thread_allreduce(
                                        T.uint32(1),
                                        c3[0],
                                        True,
                                        bo3[0],
                                        warp_id,
                                        dtype="handle",
                                    )
                                )

                            Out_partial[task_id, split_id, 0 * 32 + lane_id] = (
                                bo0[0] / bs[0]
                            ).astype(dtype)
                            Out_partial[task_id, split_id, 1 * 32 + lane_id] = (
                                bo1[0] / bs[0]
                            ).astype(dtype)
                            Out_partial[task_id, split_id, 2 * 32 + lane_id] = (
                                bo2[0] / bs[0]
                            ).astype(dtype)
                            Out_partial[task_id, split_id, 3 * 32 + lane_id] = (
                                bo3[0] / bs[0]
                            ).astype(dtype)

                            if warp_id == 0 and lane_id == 0:
                                LSE_partial[task_id, split_id] = (
                                    bm[0] * log2e + T.log2(bs[0])
                                )

            return main

        return kernel


@register_kernel("attn_decode", "flash_decode_tilelang_split")
class FlashDecodeSplitKvTileLangKernel(_FlashDecodeTileLangBase):
    """LDG-based flash decoding (split-KV variant).

    Same persistent-grid + num_split work distribution as the TMA backend,
    but each warp reads K/V directly from global memory (4x LDG.U16 per
    position) instead of going through shared-memory tiles. The inner
    dot-product uses `T.tvm_thread_allreduce` (works because there's no
    `T.Pipelined` wrapping it).
    """

    def _make_split_kernel(self):
        total_tasks = self.batch * self.num_heads
        num_split = self.num_split
        total_work_units = total_tasks * num_split
        num_iters = (total_work_units + self.num_sms - 1) // self.num_sms
        seq_len = self.seq_len
        head_dim = self.head_dim
        num_sms = self.num_sms
        kv_chunk = seq_len // num_split

        def get_configs():
            return [{"num_warps": nw} for nw in [2, 4, 8, 16]]

        @autotune(configs=get_configs(), warmup=5, rep=50)
        @tilelang.jit(target="auto")
        def kernel(num_warps=None):
            dtype = "bfloat16"
            accum_dtype = "float"
            attn_scale = 1.0 / (head_dim**0.5)
            log2e = 1.44269504
            sum_red = T.comm_reducer(lambda x, y: x + y, [T.cast(0, accum_dtype)])
            max_red = T.comm_reducer(lambda x, y: T.max(x, y), [T.cast(-1e30, accum_dtype)])

            if num_warps is None:
                num_warps = 4

            @T.prim_func
            def main(
                Q: T.Buffer((total_tasks, head_dim), dtype),
                K: T.Buffer((total_tasks, seq_len, head_dim), dtype),
                V: T.Buffer((total_tasks, seq_len, head_dim), dtype),
                Out_partial: T.Buffer((total_tasks, num_split, head_dim), dtype),
                LSE_partial: T.Buffer((total_tasks, num_split), accum_dtype),
            ):
                with T.Kernel(num_sms, threads=(32, num_warps)) as block_id:
                    lane_id = T.get_thread_binding(0)
                    warp_id = T.get_thread_binding(1)

                    Q_shared = T.alloc_shared((head_dim,), accum_dtype)

                    for it in T.serial(num_iters):
                        work_id = it * num_sms + block_id
                        if work_id < total_work_units:
                            split_id = work_id % num_split
                            task_id = work_id // num_split
                            kv_start = split_id * kv_chunk

                            for d in T.Parallel(head_dim):
                                Q_shared[d] = Q[task_id, d].astype(accum_dtype)

                            q0 = Q_shared[0 * 32 + lane_id]
                            q1 = Q_shared[1 * 32 + lane_id]
                            q2 = Q_shared[2 * 32 + lane_id]
                            q3 = Q_shared[3 * 32 + lane_id]

                            o0 = T.alloc_local((1,), accum_dtype)
                            o1 = T.alloc_local((1,), accum_dtype)
                            o2 = T.alloc_local((1,), accum_dtype)
                            o3 = T.alloc_local((1,), accum_dtype)
                            T.clear(o0)
                            T.clear(o1)
                            T.clear(o2)
                            T.clear(o3)
                            ms = T.alloc_local((1,), accum_dtype)
                            ms[0] = -1e30
                            ls = T.alloc_local((1,), accum_dtype)
                            ls[0] = 0.0

                            # Stride pattern: each warp walks positions
                            # pw*num_warps + warp_id, reading K/V via direct LDG.
                            for pw in T.serial(T.ceildiv(kv_chunk, num_warps)):
                                pos = kv_start + pw * num_warps + warp_id
                                if pos < kv_start + kv_chunk:
                                    k0 = K[
                                        task_id, pos, 0 * 32 + lane_id
                                    ].astype(accum_dtype)
                                    k1 = K[
                                        task_id, pos, 1 * 32 + lane_id
                                    ].astype(accum_dtype)
                                    k2 = K[
                                        task_id, pos, 2 * 32 + lane_id
                                    ].astype(accum_dtype)
                                    k3 = K[
                                        task_id, pos, 3 * 32 + lane_id
                                    ].astype(accum_dtype)

                                    sp = T.alloc_local((1,), accum_dtype)
                                    sp[0] = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3

                                    sc = T.alloc_local((1,), accum_dtype)
                                    with T.attr(
                                        sum_red,
                                        "reduce_scope",
                                        T.reinterpret(T.uint64(0), dtype="handle"),
                                    ):
                                        T.evaluate(
                                            T.tvm_thread_allreduce(
                                                T.uint32(1),
                                                sp[0],
                                                True,
                                                sc[0],
                                                lane_id,
                                                dtype="handle",
                                            )
                                        )
                                    sc[0] *= attn_scale

                                    om = T.alloc_local((1,), accum_dtype)
                                    om[0] = ms[0]
                                    ms[0] = T.max(ms[0], sc[0])
                                    ed = T.exp2((om[0] - ms[0]) * log2e)
                                    wt = T.exp2((sc[0] - ms[0]) * log2e)
                                    ls[0] = ls[0] * ed + wt

                                    v0 = V[
                                        task_id, pos, 0 * 32 + lane_id
                                    ].astype(accum_dtype)
                                    v1 = V[
                                        task_id, pos, 1 * 32 + lane_id
                                    ].astype(accum_dtype)
                                    v2 = V[
                                        task_id, pos, 2 * 32 + lane_id
                                    ].astype(accum_dtype)
                                    v3 = V[
                                        task_id, pos, 3 * 32 + lane_id
                                    ].astype(accum_dtype)

                                    o0[0] = o0[0] * ed + wt * v0
                                    o1[0] = o1[0] * ed + wt * v1
                                    o2[0] = o2[0] * ed + wt * v2
                                    o3[0] = o3[0] * ed + wt * v3

                            # Cross-warp reduction (inlined; the helper version
                            # trips tilelang's AST builder).
                            bm = T.alloc_local((1,), accum_dtype)
                            with T.attr(
                                max_red,
                                "reduce_scope",
                                T.reinterpret(T.uint64(0), dtype="handle"),
                            ):
                                T.evaluate(
                                    T.tvm_thread_allreduce(
                                        T.uint32(1),
                                        ms[0],
                                        True,
                                        bm[0],
                                        warp_id,
                                        dtype="handle",
                                    )
                                )

                            ws = T.exp2((ms[0] - bm[0]) * log2e)

                            cs = T.alloc_local((1,), accum_dtype)
                            cs[0] = ls[0] * ws
                            bs = T.alloc_local((1,), accum_dtype)
                            with T.attr(
                                sum_red,
                                "reduce_scope",
                                T.reinterpret(T.uint64(0), dtype="handle"),
                            ):
                                T.evaluate(
                                    T.tvm_thread_allreduce(
                                        T.uint32(1),
                                        cs[0],
                                        True,
                                        bs[0],
                                        warp_id,
                                        dtype="handle",
                                    )
                                )

                            c0 = T.alloc_local((1,), accum_dtype)
                            c0[0] = o0[0] * ws
                            bo0 = T.alloc_local((1,), accum_dtype)
                            with T.attr(
                                sum_red,
                                "reduce_scope",
                                T.reinterpret(T.uint64(0), dtype="handle"),
                            ):
                                T.evaluate(
                                    T.tvm_thread_allreduce(
                                        T.uint32(1),
                                        c0[0],
                                        True,
                                        bo0[0],
                                        warp_id,
                                        dtype="handle",
                                    )
                                )

                            c1 = T.alloc_local((1,), accum_dtype)
                            c1[0] = o1[0] * ws
                            bo1 = T.alloc_local((1,), accum_dtype)
                            with T.attr(
                                sum_red,
                                "reduce_scope",
                                T.reinterpret(T.uint64(0), dtype="handle"),
                            ):
                                T.evaluate(
                                    T.tvm_thread_allreduce(
                                        T.uint32(1),
                                        c1[0],
                                        True,
                                        bo1[0],
                                        warp_id,
                                        dtype="handle",
                                    )
                                )

                            c2 = T.alloc_local((1,), accum_dtype)
                            c2[0] = o2[0] * ws
                            bo2 = T.alloc_local((1,), accum_dtype)
                            with T.attr(
                                sum_red,
                                "reduce_scope",
                                T.reinterpret(T.uint64(0), dtype="handle"),
                            ):
                                T.evaluate(
                                    T.tvm_thread_allreduce(
                                        T.uint32(1),
                                        c2[0],
                                        True,
                                        bo2[0],
                                        warp_id,
                                        dtype="handle",
                                    )
                                )

                            c3 = T.alloc_local((1,), accum_dtype)
                            c3[0] = o3[0] * ws
                            bo3 = T.alloc_local((1,), accum_dtype)
                            with T.attr(
                                sum_red,
                                "reduce_scope",
                                T.reinterpret(T.uint64(0), dtype="handle"),
                            ):
                                T.evaluate(
                                    T.tvm_thread_allreduce(
                                        T.uint32(1),
                                        c3[0],
                                        True,
                                        bo3[0],
                                        warp_id,
                                        dtype="handle",
                                    )
                                )

                            Out_partial[task_id, split_id, 0 * 32 + lane_id] = (
                                bo0[0] / bs[0]
                            ).astype(dtype)
                            Out_partial[task_id, split_id, 1 * 32 + lane_id] = (
                                bo1[0] / bs[0]
                            ).astype(dtype)
                            Out_partial[task_id, split_id, 2 * 32 + lane_id] = (
                                bo2[0] / bs[0]
                            ).astype(dtype)
                            Out_partial[task_id, split_id, 3 * 32 + lane_id] = (
                                bo3[0] / bs[0]
                            ).astype(dtype)

                            if warp_id == 0 and lane_id == 0:
                                LSE_partial[task_id, split_id] = (
                                    bm[0] * log2e + T.log2(bs[0])
                                )

            return main

        return kernel


# ---------------------------------------------------------------------------
# GQA flash decoding, migrated from
# tilelang/examples/flash_decoding/example_gqa_decode.py
# ---------------------------------------------------------------------------


def _make_gqa_flash_decode_kernel(batch, heads, kv_heads, seq_len, dim, num_split_choices):
    """JIT factory for the GQA decode kernel, migrated from tilelang's
    examples/flash_decoding/example_gqa_decode.py.

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
                                o_accum[0] += (
                                    Output_partial[brow, hcol, s, idx].astype(accum_dtype) * w
                                )
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
            batch, num_heads, num_kv_heads, seq_len, head_dim, self._num_split_choices
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
