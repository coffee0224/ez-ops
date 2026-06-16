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
