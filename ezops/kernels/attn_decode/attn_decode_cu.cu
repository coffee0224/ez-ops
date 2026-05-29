#include <tvm/ffi/tvm_ffi.h>
#include <math.h>

// TODO: implement the CUDA kernel for attn_decode
// attn_decode computes: out = softmax(Q @ K^T / sqrt(head_dim)) @ V
// where Q is [batch, num_heads, 1, head_dim]
//       K is [batch, num_heads, seq_len, head_dim]
//       V is [batch, num_heads, seq_len, head_dim]
//       out is [batch, num_heads, 1, head_dim]
__global__ void attn_decode_kernel(
    const float* Q, const float* K, const float* V, float* Out,
    int batch, int num_heads, int seq_len, int head_dim,
    int stride_qb, int stride_qh, int stride_qm, int stride_qd,
    int stride_kb, int stride_kh, int stride_km, int stride_kd,
    int stride_vb, int stride_vh, int stride_vm, int stride_vd,
    int stride_ob, int stride_oh, int stride_om, int stride_od) {
    // TODO: kernel body
}

void attn_decode_cu(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView Out) {
    // TODO: host launch logic
    // Use TVMFFIEnvGetStream for the CUDA stream:
    //   DLDevice dev = Q.device();
    //   cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
}
