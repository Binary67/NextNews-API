import asyncio

from sqlalchemy import select

from app.agents import AGENT_NAMES
from app.ai import GeneratedPostContent
from app.config import Settings
from app.database import create_app_engine, create_session_factory, init_database
from app.models import GeneratedPost, SourceItem
from app.pipeline import (
    create_ready_generated_post,
    generate_missing_posts,
    insert_source_item,
    source_items_without_generated_post,
)


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
    assert posts[0].agent_name in AGENT_NAMES


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
        article_texts: list[str | None] = []

        def __init__(self, settings: Settings) -> None:
            pass

        async def generate_post_content(self, source_item: SourceItem) -> GeneratedPostContent:
            self.article_texts.append(source_item.article_text)
            return generated_content()

        async def generate_image(self, prompt: str) -> bytes:
            return b"image-bytes"

    async def fake_fetch_article_text(client, url: str) -> str:
        return f"Fetched article text from {url}"

    monkeypatch.setattr("app.pipeline.AzureResponsesClient", FakeAzureResponsesClient)
    monkeypatch.setattr("app.pipeline.fetch_article_text", fake_fetch_article_text)

    with session_factory() as session:
        assert insert_source_item(session, hn_item(1)) is not None
        assert insert_source_item(session, hn_item(2)) is not None

        generated_count = asyncio.run(generate_missing_posts(session, settings))

        assert generated_count == 1
        posts = session.scalars(select(GeneratedPost)).all()
        assert len(posts) == 1
        assert posts[0].status == "ready"
        assert FakeAzureResponsesClient.article_texts == [
            "Fetched article text from https://example.com/sqlite"
        ]


def test_generate_missing_posts_marks_article_fetch_failure_as_failed(
    tmp_path,
    monkeypatch,
) -> None:
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

    class UnexpectedAzureResponsesClient:
        def __init__(self, settings: Settings) -> None:
            pass

        async def generate_post_content(self, source_item: SourceItem) -> GeneratedPostContent:
            raise AssertionError("content generation should not run")

        async def generate_image(self, prompt: str) -> bytes:
            raise AssertionError("image generation should not run")

    async def failing_fetch_article_text(client, url: str) -> str:
        raise RuntimeError("blocked by upstream")

    monkeypatch.setattr("app.pipeline.AzureResponsesClient", UnexpectedAzureResponsesClient)
    monkeypatch.setattr("app.pipeline.fetch_article_text", failing_fetch_article_text)

    with session_factory() as session:
        source_item = insert_source_item(session, hn_item(1))
        assert source_item is not None

        generated_count = asyncio.run(generate_missing_posts(session, settings))

        assert generated_count == 0
        posts = session.scalars(select(GeneratedPost)).all()
        assert len(posts) == 1
        assert posts[0].source_item_id == source_item.id
        assert posts[0].status == "failed"
        assert posts[0].error_message is not None
        assert "blocked by upstream" in posts[0].error_message
        assert source_items_without_generated_post(session, 10) == []


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

    async def fake_fetch_article_text(client, url: str) -> str:
        return f"Fetched article text from {url}"

    monkeypatch.setattr("app.pipeline.fetch_article_text", fake_fetch_article_text)

    with session_factory() as session:
        source_item = insert_source_item(session, hn_item(1))
        assert source_item is not None

        monkeypatch.setattr("app.pipeline.AzureResponsesClient", FailingAzureResponsesClient)
        failed_count = asyncio.run(generate_missing_posts(session, settings))

        assert failed_count == 0
        assert session.scalars(select(GeneratedPost)).all() == []
        remaining_source_items = source_items_without_generated_post(session, 10)
        assert [item.id for item in remaining_source_items] == [source_item.id]

        monkeypatch.setattr("app.pipeline.AzureResponsesClient", SuccessfulAzureResponsesClient)
        generated_count = asyncio.run(generate_missing_posts(session, settings))

        assert generated_count == 1
        posts = session.scalars(select(GeneratedPost)).all()
        assert len(posts) == 1
        assert posts[0].status == "ready"
