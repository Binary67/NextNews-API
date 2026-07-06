from sqlalchemy import select

from app.config import Settings
from app.database import create_app_engine, create_session_factory, init_database
from app.models import GeneratedPost, SourceItem
from app.pipeline import claim_generated_post, insert_source_item


def make_session():
    settings = Settings(database_url="sqlite://", hn_pipeline_interval_seconds=60)
    engine = create_app_engine(settings.database_url)
    init_database(engine, settings.database_url)
    session_factory = create_session_factory(engine)
    return settings, session_factory()


def hn_item(item_id: int) -> dict:
    return {
        "id": item_id,
        "type": "story",
        "title": "SQLite for local apps",
        "url": "https://example.com/sqlite",
        "by": "author",
        "score": 42,
    }


def test_insert_source_item_dedupes_by_source_and_hn_id() -> None:
    _, session = make_session()

    first = insert_source_item(session, hn_item(123))
    second = insert_source_item(session, hn_item(123))

    assert first is not None
    assert second is None
    assert len(session.scalars(select(SourceItem)).all()) == 1


def test_claim_generated_post_dedupes_by_source_item() -> None:
    settings, session = make_session()
    source_item = insert_source_item(session, hn_item(456))
    assert source_item is not None

    first = claim_generated_post(session, source_item, settings)
    second = claim_generated_post(session, source_item, settings)

    assert first is not None
    assert second is None
    posts = session.scalars(select(GeneratedPost)).all()
    assert len(posts) == 1
    assert posts[0].status == "processing"

