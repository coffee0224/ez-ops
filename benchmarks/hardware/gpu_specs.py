"""GPU specifications database.

Theoretical HBM bandwidth and tensor core peak FLOPS (dense, no sparsity).
All TFLOPS values use FMA=2 counting (multiply + add counted separately),
matching the standard roofline convention (flops = 2*M*N*K for matmul).

Convention:
  - FP16/BF16/TF32: datacenter uses NVIDIA headline (FP16 acc for Hopper/Ada-DC);
    consumer uses FP32 acc (from NVIDIA Ada/Blackwell whitepapers).
  - FP8: all GPUs use FP16 accumulate, matching torch._scaled_mm behavior.
    FP8 = 2 × FP8(FP32 acc) = 4 × FP16(FP32 acc) = 8 × FP32.
"""

# Profile key → specs
# Each entry: names (from nvidia-smi / torch.cuda), compute_cap, hbm_bw_gb,
# tensor_core_tflops (TFLOPS, dense, FMA=2).
PROFILES = {
    # ── Hopper ────────────────────────────────────────────────────────────
    "h200": {
        "names": ["NVIDIA H200"],
        "compute_cap": "9.0",
        "hbm_bw_gb": 4800.0,
        "tensor_core_tflops": {
            "fp16": 1979.0, "bf16": 1979.0,
            "tf32": 989.0, "fp8": 3958.0,
        },
    },
    "h100_sxm": {
        "names": ["NVIDIA H100 SXM", "NVIDIA H100 SXM5"],
        "compute_cap": "9.0",
        "hbm_bw_gb": 3352.0,
        "tensor_core_tflops": {
            "fp16": 1979.0, "bf16": 1979.0,
            "tf32": 989.0, "fp8": 3958.0,
        },
    },
    "h100_pcie": {
        "names": ["NVIDIA H100 PCIe"],
        "compute_cap": "9.0",
        "hbm_bw_gb": 2039.0,
        "tensor_core_tflops": {
            "fp16": 1513.0, "bf16": 1513.0,
            "tf32": 756.0, "fp8": 3026.0,
        },
    },
    # ── Ampere ────────────────────────────────────────────────────────────
    "a100_sxm_80gb": {
        "names": ["NVIDIA A100-SXM4-80GB"],
        "compute_cap": "8.0",
        "hbm_bw_gb": 2039.0,
        "tensor_core_tflops": {
            "fp16": 312.0, "bf16": 312.0, "tf32": 156.0,
        },
    },
    "a100_sxm_40gb": {
        "names": ["NVIDIA A100-SXM4-40GB", "NVIDIA A100-SXM4"],
        "compute_cap": "8.0",
        "hbm_bw_gb": 1555.0,
        "tensor_core_tflops": {
            "fp16": 312.0, "bf16": 312.0, "tf32": 156.0,
        },
    },
    "a100_pcie_80gb": {
        "names": ["NVIDIA A100-PCIe-80GB"],
        "compute_cap": "8.0",
        "hbm_bw_gb": 2039.0,
        "tensor_core_tflops": {
            "fp16": 312.0, "bf16": 312.0, "tf32": 156.0,
        },
    },
    "a100_pcie_40gb": {
        "names": ["NVIDIA A100-PCIe-40GB", "NVIDIA A100-PCIe"],
        "compute_cap": "8.0",
        "hbm_bw_gb": 1555.0,
        "tensor_core_tflops": {
            "fp16": 312.0, "bf16": 312.0, "tf32": 156.0,
        },
    },
    # ── Ada Lovelace (datacenter) ─────────────────────────────────────────
    "l40s": {
        "names": ["NVIDIA L40S"],
        "compute_cap": "8.9",
        "hbm_bw_gb": 864.0,
        "tensor_core_tflops": {
            "fp16": 366.0, "bf16": 366.0,
            "tf32": 183.0, "fp8": 733.0,
        },
    },
    "l40": {
        "names": ["NVIDIA L40"],
        "compute_cap": "8.9",
        "hbm_bw_gb": 864.0,
        "tensor_core_tflops": {
            "fp16": 362.0, "bf16": 362.0,
            "tf32": 181.0, "fp8": 724.0,
        },
    },
    # ── Ada Lovelace (consumer) — FP16/BF16/TF32: FP32 acc; FP8: FP16 acc ──
    # Source: NVIDIA Ada Architecture Whitepaper
    # FP16/BF16(FP32 acc) = 2×FP32, TF32 = FP32, FP8(FP16 acc) = 8×FP32
    "rtx_4090": {
        "names": ["NVIDIA GeForce RTX 4090"],
        "compute_cap": "8.9",
        "hbm_bw_gb": 1008.0,
        "tensor_core_tflops": {
            "fp16": 165.2, "bf16": 165.2,
            "tf32": 82.6, "fp8": 660.8,
        },
    },
    "rtx_4080": {
        "names": ["NVIDIA GeForce RTX 4080"],
        "compute_cap": "8.9",
        "hbm_bw_gb": 716.8,
        "tensor_core_tflops": {
            "fp16": 97.5, "bf16": 97.5,
            "tf32": 48.7, "fp8": 389.6,
        },
    },
    "rtx_4070_ti": {
        "names": ["NVIDIA GeForce RTX 4070 Ti"],
        "compute_cap": "8.9",
        "hbm_bw_gb": 504.0,
        "tensor_core_tflops": {
            "fp16": 80.2, "bf16": 80.2,
            "tf32": 40.1, "fp8": 320.8,
        },
    },
    "rtx_4070": {
        "names": ["NVIDIA GeForce RTX 4070"],
        "compute_cap": "8.9",
        "hbm_bw_gb": 504.0,
        "tensor_core_tflops": {
            "fp16": 58.3, "bf16": 58.3,
            "tf32": 29.1, "fp8": 232.8,
        },
    },
    # ── Blackwell (consumer) — FP16/BF16/TF32: FP32 acc; FP8: FP16 acc ────
    # Source: NVIDIA RTX Blackwell GPU Architecture Whitepaper
    # FP16(FP32 acc) = FP16(FP16 acc)/2; FP8(FP16 acc) = 4×FP16(FP32 acc) = 8×FP32
    "rtx_5090": {
        "names": ["NVIDIA GeForce RTX 5090"],
        "compute_cap": "12.0",
        "hbm_bw_gb": 1792.0,
        "tensor_core_tflops": {
            "fp16": 209.5, "bf16": 209.5,
            "tf32": 104.8, "fp8": 838.0,
        },
    },
    "rtx_5080": {
        "names": ["NVIDIA GeForce RTX 5080"],
        "compute_cap": "12.0",
        "hbm_bw_gb": 960.0,
        "tensor_core_tflops": {
            "fp16": 112.6, "bf16": 112.6,
            "tf32": 56.3, "fp8": 450.2,
        },
    },
    "rtx_5070_ti": {
        "names": ["NVIDIA GeForce RTX 5070 Ti"],
        "compute_cap": "12.0",
        "hbm_bw_gb": 896.0,
        "tensor_core_tflops": {
            "fp16": 87.9, "bf16": 87.9,
            "tf32": 43.9, "fp8": 351.6,
        },
    },
    "rtx_5070": {
        "names": ["NVIDIA GeForce RTX 5070"],
        "compute_cap": "12.0",
        "hbm_bw_gb": 672.0,
        "tensor_core_tflops": {
            "fp16": 61.8, "bf16": 61.8,
            "tf32": 30.9, "fp8": 247.0,
        },
    },
    "rtx_5060_ti": {
        "names": ["NVIDIA GeForce RTX 5060 Ti"],
        "compute_cap": "12.0",
        "hbm_bw_gb": 448.0,
        "tensor_core_tflops": {
            "fp16": 47.4, "bf16": 47.4,
            "tf32": 23.7, "fp8": 189.6,
        },
    },
    "rtx_5060": {
        "names": ["NVIDIA GeForce RTX 5060"],
        "compute_cap": "12.0",
        "hbm_bw_gb": 448.0,
        "tensor_core_tflops": {
            "fp16": 39.5, "bf16": 39.5,
            "tf32": 19.8, "fp8": 158.0,
        },
    },
}

# Reverse mapping: name → profile key
_NAME_MAP: dict[str, str] = {}
for _key, _spec in PROFILES.items():
    for _name in _spec["names"]:
        _NAME_MAP[_name] = _key


def detect_profile(gpu_name: str) -> str | None:
    """Match a GPU name string to a profile key.

    Tries exact match first, then prefix match to handle variants
    like torch.cuda's "NVIDIA H200 141GB HBM3e".
    """
    if gpu_name in _NAME_MAP:
        return _NAME_MAP[gpu_name]
    # Prefix: input starts with a known name
    for name, key in _NAME_MAP.items():
        if gpu_name.startswith(name):
            return key
    # Reverse prefix: known name starts with the input
    for name, key in _NAME_MAP.items():
        if name.startswith(gpu_name):
            return key
    return None


def get_specs(profile: str) -> dict | None:
    """Get specs for a profile key."""
    return PROFILES.get(profile)
