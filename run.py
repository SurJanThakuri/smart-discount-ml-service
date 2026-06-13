import sys

from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run.py [train|serve]")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "train":
        from src.models.train import train_pipeline

        train_pipeline()
    elif command == "serve":
        from app import app as flask_app
        import config

        logger.info(
            "Starting Flask server on %s:%s", config.FLASK_HOST, config.FLASK_PORT
        )
        flask_app.run(
            host=config.FLASK_HOST, port=config.FLASK_PORT, debug=False
        )
    else:
        print(f"Unknown command: {command}")
        print("Usage: python run.py [train|serve]")
        sys.exit(1)


if __name__ == "__main__":
    main()
