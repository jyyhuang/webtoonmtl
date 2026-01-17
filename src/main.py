import appdirs
import atexit
import json
import logging
import logging.config
import pathlib

logger = logging.getLogger(__name__)


def setup_logging():
    config_file = pathlib.Path("config/logging_config.json")
    log_dir = pathlib.Path(appdirs.user_log_dir("webtoonmtl"))
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(config_file) as file:
        config = json.load(file)

    config["handlers"]["json_file"]["filename"] = str(
        log_dir / "webtoonmtl.log.jsonl"
    )

    logging.config.dictConfig(config)

    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)
    

def main():
    setup_logging()


if __name__ == "__main__":
    main()
