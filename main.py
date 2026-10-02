import argparse
import logging
from dataclasses import replace
from face_app.application import FaceIdentificationApp
from face_app.backend import configure_environment, resolve_device
from face_app.config import AppConfig
from face_app.logging_config import configure_logging


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Value must be at least 1.")
    return number


def parse_args() -> argparse.Namespace:
    defaults = AppConfig()
    parser = argparse.ArgumentParser(description="Webcam face identification on CPU or NVIDIA GPU.")
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default=defaults.yolo_device,
                        help="cpu (default), gpu (requires CUDA), or auto (GPU with CPU fallback)")
    parser.add_argument("--camera", type=int, default=defaults.camera_index, help="Webcam index (default: 0)")
    parser.add_argument("--every-n-frames", type=positive_int, default=defaults.process_every_n_frames,
                        help="Run detection/recognition every N frames (default: 5)")
    parser.add_argument("--check-device", action="store_true",
                        help="Validate and print the selected device without loading models or opening a camera")
    return parser.parse_args()


def main() -> None:
    """Application entry point."""
    args = parse_args()
    configure_logging()
    configure_environment(args.device)

    try:
        device = resolve_device(args.device)
    except (ImportError, RuntimeError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        raise SystemExit(1) from exc

    if args.check_device:
        return

    config = replace(
        AppConfig(),
        yolo_device=device,
        camera_index=args.camera,
        process_every_n_frames=args.every_n_frames,
    )
    app = FaceIdentificationApp(config)

    try:
        app.run()
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Stopped by user.")


if __name__ == "__main__":
    main()
