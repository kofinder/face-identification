from face_app.application import FaceIdentificationApp
from face_app.config import AppConfig
from face_app.logging_config import configure_logging

def main() -> None:
    """Application entry point."""
    configure_logging()

    config = AppConfig()
    app = FaceIdentificationApp(config)
    app.run()


if __name__ == "__main__":
    main()