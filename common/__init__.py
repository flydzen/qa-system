import logging
from queue import Queue
from os import getenv

from logging_loki import LokiQueueHandler


def setup_logging(app_name: str) -> logging.Logger:
    loki_logs_handler = LokiQueueHandler(
        Queue(-1),
        url=getenv('LOKI_ENDPOINT') or 'http://127.0.0.1:3100/loki/api/v1/push',
        tags={'application': app_name},
        version="1",
    )
    file_handler = logging.FileHandler('app_logs.txt')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_logger = logging.getLogger('uvicorn')
    app_logger = logging.getLogger('app')
    app_logger.setLevel(logging.INFO)

    for _logger in (uvicorn_access_logger, uvicorn_logger, app_logger):
        _logger.addHandler(loki_logs_handler)
        _logger.addHandler(file_handler)

    return app_logger
