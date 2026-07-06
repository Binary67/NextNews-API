from pathlib import Path
from typing import Iterator
from urllib.parse import unquote, urlparse

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


def _sqlite_file_path(database_url: str) -> Path | None:
    parsed = urlparse(database_url)
    if parsed.scheme != "sqlite":
        return None

    if parsed.path in ("", "/"):
        return None

    if parsed.path == "/:memory:":
        return None

    if database_url.startswith("sqlite:///./"):
        return Path(unquote(database_url.removeprefix("sqlite:///")))

    return Path(unquote(parsed.path))


def create_app_engine(database_url: str) -> Engine:
    connect_args = {}
    engine_kwargs = {}

    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    if database_url in {"sqlite://", "sqlite:///:memory:"}:
        engine_kwargs["poolclass"] = StaticPool

    return create_engine(database_url, connect_args=connect_args, **engine_kwargs)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_database(engine: Engine, database_url: str) -> None:
    db_path = _sqlite_file_path(database_url)
    if db_path is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
    finally:
        session.close()

