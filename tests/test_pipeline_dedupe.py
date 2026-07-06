import asyncio

from sqlalchemy import select

from app.ai import GeneratedPostContent
from app.config import Settings
from app.database import create_app_engine, create_session_factory, init_database
from app.models import GeneratedPost, SourceItem
from app.pipeline import create_ready_generated_post, generate_missing_posts, insert_source_item


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


def generated_content() -> GeneratedPostContent:
    return GeneratedPostContent(
        title="Generated title",
        description="Generated description",
        content="Generated content",
        image_prompt="Generated image prompt",
    )


def test_create_ready_generated_post_dedupes_by_source_item(tmp_path) -> None:
    settings, session = make_session()
    source_item = insert_source_item(session, hn_item(456))
    assert source_item is not None

    first = create_ready_generated_post(
        session,
        source_item,
        settings,
        generated_content(),
        b"image-bytes",
        tmp_path,
    )
    second = create_ready_generated_post(
        session,
        source_item,
        settings,
        generated_content(),
        b"image-bytes",
        tmp_path,
    )

    assert first is not None
    assert second is None
    posts = session.scalars(select(GeneratedPost)).all()
    assert len(posts) == 1
    assert posts[0].status == "ready"


def test_generate_missing_posts_respects_generation_limit(tmp_path, monkeypatch) -> None:
    settings = Settings(
        database_url="sqlite://",
        hn_pipeline_interval_seconds=60,
        post_generation_limit=1,
        image_output_dir=str(tmp_path / "images"),
        azure_openai_llm_endpoint="https://example.openai.azure.com",
        azure_openai_llm_api_key="llm-key",
        azure_openai_llm_deployment="llm",
        azure_openai_image_endpoint="https://example.cognitiveservices.azure.com",
        azure_openai_image_api_key="image-key",
        azure_openai_image_deployment="image",
    )
    engine = create_app_engine(settings.database_url)
    init_database(engine, settings.database_url)
    session_factory = create_session_factory(engine)

    class FakeAzureResponsesClient:
        def __init__(self, settings: Settings) -> None:
            pass

        async def generate_post_content(self, source_item: SourceItem) -> GeneratedPostContent:
            return generated_content()

        async def generate_image(self, prompt: str) -> bytes:
            return b"image-bytes"

    monkeypatch.setattr("app.pipeline.AzureResponsesClient", FakeAzureResponsesClient)

    with session_factory() as session:
        assert insert_source_item(session, hn_item(1)) is not None
        assert insert_source_item(session, hn_item(2)) is not None

        generated_count = asyncio.run(generate_missing_posts(session, settings))

        assert generated_count == 1
        posts = session.scalars(select(GeneratedPost)).all()
        assert len(posts) == 1
        assert posts[0].status == "ready"


def test_generate_missing_posts_retries_after_failed_generation(tmp_path, monkeypatch) -> None:
    settings = Settings(
        database_url="sqlite://",
        hn_pipeline_interval_seconds=60,
        image_output_dir=str(tmp_path / "images"),
        azure_openai_llm_endpoint="https://example.openai.azure.com",
        azure_openai_llm_api_key="llm-key",
        azure_openai_llm_deployment="llm",
        azure_openai_image_endpoint="https://example.cognitiveservices.azure.com",
        azure_openai_image_api_key="image-key",
        azure_openai_image_deployment="image",
    )
    engine = create_app_engine(settings.database_url)
    init_database(engine, settings.database_url)
    session_factory = create_session_factory(engine)

    class FailingAzureResponsesClient:
        def __init__(self, settings: Settings) -> None:
            pass

        async def generate_post_content(self, source_item: SourceItem) -> GeneratedPostContent:
            raise RuntimeError("temporary failure")

        async def generate_image(self, prompt: str) -> bytes:
            raise AssertionError("image generation should not run")

    class SuccessfulAzureResponsesClient:
        def __init__(self, settings: Settings) -> None:
            pass

        async def generate_post_content(self, source_item: SourceItem) -> GeneratedPostContent:
            return generated_content()

        async def generate_image(self, prompt: str) -> bytes:
            return b"image-bytes"

    with session_factory() as session:
        assert insert_source_item(session, hn_item(1)) is not None

        monkeypatch.setattr("app.pipeline.AzureResponsesClient", FailingAzureResponsesClient)
        failed_count = asyncio.run(generate_missing_posts(session, settings))

        assert failed_count == 0
        assert session.scalars(select(GeneratedPost)).all() == []

        monkeypatch.setattr("app.pipeline.AzureResponsesClient", SuccessfulAzureResponsesClient)
        generated_count = asyncio.run(generate_missing_posts(session, settings))

        assert generated_count == 1
        posts = session.scalars(select(GeneratedPost)).all()
        assert len(posts) == 1
        assert posts[0].status == "ready"
