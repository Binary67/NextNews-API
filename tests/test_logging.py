import logging
import re
from pathlib import Path

import pytest

from app.main import _prune_old_logs, configure_app_logging


def _clear_app_logger_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


@pytest.fixture
def app_logger() -> logging.Logger:
    logger = logging.getLogger("app")
    _clear_app_logger_handlers(logger)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True

    yield logger

    _clear_app_logger_handlers(logger)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def test_configure_app_logging_creates_timestamped_file_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    app_logger: logging.Logger,
) -> None:
    monkeypatch.chdir(tmp_path)

    configure_app_logging()

    file_handlers = [
        handler
        for handler in app_logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]
    assert len(file_handlers) == 1
    log_path = Path(file_handlers[0].baseFilename)
    assert log_path.parent == tmp_path / "logs"
    assert re.fullmatch(r"\d{8}T\d{6}Z\.log", log_path.name)
    assert log_path.exists()
    assert app_logger.level == logging.INFO
    assert app_logger.propagate is False


def test_configure_app_logging_replaces_stream_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    app_logger: logging.Logger,
) -> None:
    monkeypatch.chdir(tmp_path)
    stream_handler = logging.StreamHandler()
    app_logger.addHandler(stream_handler)

    configure_app_logging()

    assert stream_handler not in app_logger.handlers
    assert len(app_logger.handlers) == 1
    assert isinstance(app_logger.handlers[0], logging.FileHandler)


def test_prune_old_logs_keeps_newest_three_logs(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    for log_name in [
        "20260706T010000Z.log",
        "20260706T020000Z.log",
        "20260706T030000Z.log",
        "20260706T040000Z.log",
    ]:
        (log_dir / log_name).write_text("log\n", encoding="utf-8")
    unrelated_file = log_dir / "notes.txt"
    unrelated_file.write_text("keep\n", encoding="utf-8")

    _prune_old_logs(log_dir)

    assert [path.name for path in sorted(log_dir.glob("*.log"))] == [
        "20260706T020000Z.log",
        "20260706T030000Z.log",
        "20260706T040000Z.log",
    ]
    assert unrelated_file.exists()


def test_configure_app_logging_does_not_attach_duplicate_handlers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    app_logger: logging.Logger,
) -> None:
    monkeypatch.chdir(tmp_path)

    configure_app_logging()
    first_handlers = list(app_logger.handlers)
    configure_app_logging()

    assert app_logger.handlers == first_handlers
    assert len(app_logger.handlers) == 1
    assert len(list((tmp_path / "logs").glob("*.log"))) == 1
