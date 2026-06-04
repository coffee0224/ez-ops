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


@register_kernel("attn_decode", "flash_decode_tilelang")
class FlashDecodeAttnTileLangKernel(BaseKernel):
    def __init__(self, batch, num_heads, seq_len, head_dim):
        self.batch = batch
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.num_sms = get_num_sms()
        self._kernel = self._make_kernel()
        self._best_kernel = None

    def _make_kernel(self):
        total_tasks = self.batch * self.num_heads
        num_iters = (total_tasks + self.num_sms - 1) // self.num_sms
        seq_len = self.seq_len
        head_dim = self.head_dim
        num_sms = self.num_sms

        def get_configs():
            return [{"num_warps": nw} for nw in [2, 4, 8, 16]]

        @autotune(configs=get_configs(), warmup=5, rep=50)
        @tilelang.jit(out_idx=[3], target="auto")
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
                Out: T.Buffer((total_tasks, head_dim), dtype),
            ):
                with T.Kernel(num_sms, threads=(32, num_warps)) as block_id:
                    lane_id = T.get_thread_binding(0)
                    warp_id = T.get_thread_binding(1)

                    # float32 shared memory: each 4-byte element maps to a unique bank,
                    # eliminating 2-way bank conflicts from bfloat16 (2-byte) packing.
                    # 128 x float32 = 512 bytes shared memory — negligible.
                    Q_shared = T.alloc_shared((head_dim,), accum_dtype)

                    for it in T.serial(num_iters):
                        task_id = it * num_sms + block_id
                        if task_id < total_tasks:
                            # Load Q cooperatively: 128 threads, 4 coalesced warp loads
                            for d in T.Parallel(head_dim):
                                Q_shared[d] = Q[task_id, d].astype(accum_dtype)

                            # Load Q into registers with contiguous indexing (no bank conflicts)
                            q0 = Q_shared[0 * 32 + lane_id]
                            q1 = Q_shared[1 * 32 + lane_id]
                            q2 = Q_shared[2 * 32 + lane_id]
                            q3 = Q_shared[3 * 32 + lane_id]

                            # Per-warp accumulators
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

                            # Each warp iterates positions with stride num_warps
                            for pw in T.serial(T.ceildiv(seq_len, num_warps)):
                                pos = pw * num_warps + warp_id
                                if pos < seq_len:
                                    # Coalesced K loads: 4 contiguous 32-element warp loads
                                    # instead of 4 strided (lane_id*4+j) loads hitting 2 segments each
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

                                    # Coalesced V loads: same contiguous pattern
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

                            # --- Cross-warp reduction ---
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

                            # Normalize and write output with contiguous indexing
                            Out[task_id, 0 * 32 + lane_id] = (bo0[0] / bs[0]).astype(dtype)
                            Out[task_id, 1 * 32 + lane_id] = (bo1[0] / bs[0]).astype(dtype)
                            Out[task_id, 2 * 32 + lane_id] = (bo2[0] / bs[0]).astype(dtype)
                            Out[task_id, 3 * 32 + lane_id] = (bo3[0] / bs[0]).astype(dtype)

            return main

        return kernel

    def __call__(self, Q, K, V):
        B, H, _, D = Q.shape
        S = K.shape[2]
        Q_flat = Q.reshape(B * H, D)
        K_flat = K.reshape(B * H, S, D)
        V_flat = V.reshape(B * H, S, D)

        if self._best_kernel is None:
            self._best_kernel = self._kernel()

        Out_flat = self._best_kernel(Q_flat, K_flat, V_flat)
        return Out_flat.reshape(B, H, 1, D)
