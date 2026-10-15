"""saepipe/backend.py — which GPU stack are we on? [generic]"""
import torch


def vendor():
    """'rocm' (AMD, torch built with HIP), 'cuda' (NVIDIA) or 'cpu'."""
    if getattr(torch.version, "hip", None):
        return "rocm"
    if torch.version.cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def check_bf16(device=None):
    """bf16 tl.dot needs AMD CDNA2+ (MI200/MI300) or NVIDIA Ampere+ (sm_80+)."""
    v = vendor()
    if v == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        if major < 8:
            raise RuntimeError(f"bf16 Triton dot needs an Ampere-or-newer NVIDIA GPU (sm_80+), got sm_{major}x")
    return v


def gpu_name(device=0):
    return torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu"
