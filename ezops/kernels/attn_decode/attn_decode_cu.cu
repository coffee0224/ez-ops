#include <tvm/ffi/tvm_ffi.h>
#include <math.h>
#include <cuda_bf16.h>

constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Each block handles one (batch, head). Multiple warps iterate over the KV
// sequence in parallel, accumulating an online softmax. Final output written
// directly to Out — no global workspace, no grid sync.
__global__ void __launch_bounds__(256, 1)
attn_decode_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ Out,
    int batch, int num_heads, int seq_len, int head_dim,
    int stride_qb, int stride_qh, int stride_qm, int stride_qd,
    int stride_kb, int stride_kh, int stride_km, int stride_kd,
    int stride_vb, int stride_vh, int stride_vm, int stride_vd,
    int stride_ob, int stride_oh, int stride_om, int stride_od) {
    int block_id = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    int b = block_id / num_heads;
    int h = block_id % num_heads;

    const __nv_bfloat16* Q_bh = Q + b * stride_qb + h * stride_qh;
    const __nv_bfloat16* K_bh = K + b * stride_kb + h * stride_kh;
    const __nv_bfloat16* V_bh = V + b * stride_vb + h * stride_vh;

    // Load Q head into registers (vec4)
    float q_local[4] = {};
    if (lane_id * 4 < head_dim) {
        uint2 q_u2 = __ldg(reinterpret_cast<const uint2*>(Q_bh + lane_id * 4));
        __nv_bfloat16* qp = reinterpret_cast<__nv_bfloat16*>(&q_u2);
        q_local[0] = __bfloat162float(qp[0]);
        q_local[1] = __bfloat162float(qp[1]);
        q_local[2] = __bfloat162float(qp[2]);
        q_local[3] = __bfloat162float(qp[3]);
    }

    float attn_scale = 1.0f / sqrtf((float)head_dim);

    __shared__ float s_max_score[8];
    __shared__ float s_sum_exp[8];
    __shared__ float s_out_acc[8][128];

    // Online softmax: each warp iterates over its share of positions
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int pos = warp_id; pos < seq_len; pos += num_warps) {
        const __nv_bfloat16* k_pos = K_bh + pos * stride_km;
        const __nv_bfloat16* v_pos = V_bh + pos * stride_vm;

        float score = 0.0f;
        if (lane_id * 4 < head_dim) {
            uint2 k_u2 = __ldg(reinterpret_cast<const uint2*>(k_pos + lane_id * 4));
            __nv_bfloat16* kp = reinterpret_cast<__nv_bfloat16*>(&k_u2);
            score = q_local[0] * __bfloat162float(kp[0]) +
                    q_local[1] * __bfloat162float(kp[1]) +
                    q_local[2] * __bfloat162float(kp[2]) +
                    q_local[3] * __bfloat162float(kp[3]);
        }
        score = warp_reduce_sum(score) * attn_scale;
        score = __shfl_sync(0xffffffff, score, 0);

        float old_max = max_score;
        max_score = fmaxf(max_score, score);
        float exp_diff = expf(old_max - max_score);
        sum_exp = sum_exp * exp_diff + expf(score - max_score);
        float weight = expf(score - max_score);

        if (lane_id * 4 < head_dim) {
            uint2 v_u2 = __ldg(reinterpret_cast<const uint2*>(v_pos + lane_id * 4));
            __nv_bfloat16* vp = reinterpret_cast<__nv_bfloat16*>(&v_u2);
            out_acc[0] = out_acc[0] * exp_diff + weight * __bfloat162float(vp[0]);
            out_acc[1] = out_acc[1] * exp_diff + weight * __bfloat162float(vp[1]);
            out_acc[2] = out_acc[2] * exp_diff + weight * __bfloat162float(vp[2]);
            out_acc[3] = out_acc[3] * exp_diff + weight * __bfloat162float(vp[3]);
        }
    }

    // Block-internal warp reduction
    if (lane_id == 0) {
        s_max_score[warp_id] = max_score;
        s_sum_exp[warp_id] = sum_exp;
    }
    if (lane_id * 4 + 3 < head_dim) {
        s_out_acc[warp_id][lane_id * 4 + 0] = out_acc[0];
        s_out_acc[warp_id][lane_id * 4 + 1] = out_acc[1];
        s_out_acc[warp_id][lane_id * 4 + 2] = out_acc[2];
        s_out_acc[warp_id][lane_id * 4 + 3] = out_acc[3];
    }
    __syncthreads();

    // Warp 0 combines per-warp results -> final output
    if (warp_id == 0) {
        float block_max = -INFINITY;
        for (int w = 0; w < num_warps; w++)
            if (s_max_score[w] > block_max)
                block_max = s_max_score[w];

        float block_sum = 0.0f;
        float block_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int w = 0; w < num_warps; w++) {
            if (s_max_score[w] > -INFINITY) {
                float sc = expf(s_max_score[w] - block_max);
                block_sum += s_sum_exp[w] * sc;
                int base = lane_id * 4;
                block_out[0] += s_out_acc[w][base + 0] * sc;
                block_out[1] += s_out_acc[w][base + 1] * sc;
                block_out[2] += s_out_acc[w][base + 2] * sc;
                block_out[3] += s_out_acc[w][base + 3] * sc;
            }
        }

        __nv_bfloat16* out_head = Out + b * stride_ob + h * stride_oh;
        int base = lane_id * 4;
        if (base + 3 < head_dim) {
            out_head[base + 0] = __float2bfloat16(block_out[0] / block_sum);
            out_head[base + 1] = __float2bfloat16(block_out[1] / block_sum);
            out_head[base + 2] = __float2bfloat16(block_out[2] / block_sum);
            out_head[base + 3] = __float2bfloat16(block_out[3] / block_sum);
        }
    }
}

void attn_decode_cu(
    tvm::ffi::TensorView Q, tvm::ffi::TensorView K,
    tvm::ffi::TensorView V, tvm::ffi::TensorView Out) {
    int batch = (int)Q.size(0);
    int num_heads = (int)Q.size(1);
    int seq_len = (int)K.size(2);
    int head_dim = (int)Q.size(3);

    int stride_qb = (int)Q.stride(0);
    int stride_qh = (int)Q.stride(1);
    int stride_qm = (int)Q.stride(2);
    int stride_qd = (int)Q.stride(3);
    int stride_kb = (int)K.stride(0);
    int stride_kh = (int)K.stride(1);
    int stride_km = (int)K.stride(2);
    int stride_kd = (int)K.stride(3);
    int stride_vb = (int)V.stride(0);
    int stride_vh = (int)V.stride(1);
    int stride_vm = (int)V.stride(2);
    int stride_vd = (int)V.stride(3);
    int stride_ob = (int)Out.stride(0);
    int stride_oh = (int)Out.stride(1);
    int stride_om = (int)Out.stride(2);
    int stride_od = (int)Out.stride(3);

    DLDevice dev = Q.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    int num_blocks = batch * num_heads;

    attn_decode_kernel<<<num_blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(Out.data_ptr()),
        batch, num_heads, seq_len, head_dim,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_km, stride_kd,
        stride_vb, stride_vh, stride_vm, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od);
}
