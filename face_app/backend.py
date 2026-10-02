"""Select the shared PyTorch backend and validate CPU/CUDA execution."""

import logging
import os

# Both YOLO and FaceNet use PyTorch. Set this before importing DeepFace.
os.environ["DEEPFACE_BACKEND_ENGINE"] = "pytorch"
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"

LOGGER = logging.getLogger(__name__)


def configure_environment(requested_device: str) -> None:
    """Hide CUDA for CPU mode before any AI library initializes it."""
    if requested_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def resolve_device(requested_device: str) -> str:
    """Resolve cpu/gpu/auto; explicit GPU requests must never fall back silently."""
    if requested_device not in {"cpu", "gpu", "auto"}:
        raise ValueError("Device must be cpu, gpu, or auto.")

    import torch

    LOGGER.info("PyTorch %s; CUDA runtime: %s", torch.__version__, torch.version.cuda)
    if requested_device == "cpu":
        LOGGER.info("YOLO and FaceNet device: CPU")
        return "cpu"

    if not torch.cuda.is_available():
        if requested_device == "gpu":
            raise RuntimeError(
                "GPU requested, but PyTorch cannot use CUDA. Check nvidia-smi and "
                "install a CUDA-enabled PyTorch build (see README.md). "
                "Use --device cpu to run without a GPU."
            )
        LOGGER.info("CUDA unavailable; auto mode selected CPU.")
        return "cpu"

    # Availability alone does not prove that this wheel supports the GPU's
    # architecture. Execute a small CUDA kernel before loading model weights.
    try:
        probe = torch.ones((2, 2), device="cuda:0")
        _ = probe @ probe
        torch.cuda.synchronize()
    except Exception as exc:
        if requested_device == "gpu":
            raise RuntimeError(
                "CUDA is visible but a GPU operation failed. Check the NVIDIA "
                "driver and PyTorch build compatibility (see README.md)."
            ) from exc
        LOGGER.warning("CUDA probe failed; auto mode selected CPU: %s", exc)
        return "cpu"

    LOGGER.info("YOLO and FaceNet device: cuda:0 (%s)", torch.cuda.get_device_name(0))
    return "cuda:0"
