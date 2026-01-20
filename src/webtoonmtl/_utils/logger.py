import datetime as dt
import json
import logging
from typing import override

import appdirs
import atexit
import logging.config
import pathlib

LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JSONFormatter(logging.Formatter):
    def __init__(self, *, fmt_keys: dict[str, str] | None = None):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        message = self._make_log_dict(record)
        return json.dumps(message, default=str)

    def _make_log_dict(self, record: logging.LogRecord):
        always = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),
        }

        if record.exc_info is not None:
            always["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info is not None:
            always["stack_info"] = self.formatStack(record.stack_info)
        message = {
            key: (
                msg_val
                if (msg_val := always.pop(val, None)) is not None
                else getattr(record, val)
            )
            for key, val in self.fmt_keys.items()
        }
        message.update(always)
        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message


class StdoutFilter(logging.Filter):
    @override
    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        return record.levelno <= logging.INFO


def setup_logging():
    config_file = pathlib.Path("config/logging_config.json")
    log_dir = pathlib.Path(appdirs.user_log_dir("webtoonmtl"))
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(config_file) as file:
        config = json.load(file)

    config["handlers"]["json_file"]["filename"] = str(log_dir / "webtoonmtl.log.jsonl")

    logging.config.dictConfig(config)

    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)
