import asyncio
from datetime import timedelta

from sqlalchemy import select

from app.agents import AGENT_NAMES
from app.ai import GeneratedPostContent, SourceQualityEvaluation
from app.config import Settings
from app.database import create_app_engine, create_session_factory, init_database
from app.models import GeneratedPost, SourceItem, utc_now
from app.pipeline import (
    create_ready_generated_post,
    generate_missing_posts,
    ingest_hacker_news,
    insert_source_item,
    should_refresh_hacker_news,
    source_items_without_generated_post,
)


def make_session():
    settings = Settings(database_url="sqlite://", post_generation_interval_seconds=60)
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


def test_source_items_without_generated_post_preserves_insert_order() -> None:
    _, session = make_session()

    assert insert_source_item(session, hn_item(1)) is not None
    assert insert_source_item(session, hn_item(2)) is not None

    source_items = source_items_without_generated_post(session, 10)

    assert [item.source_item_id for item in source_items] == ["1", "2"]


def test_ingest_hacker_news_skips_existing_ids_before_fetching_items(monkeypatch) -> None:
    settings = Settings(
        database_url="sqlite://",
        hn_story_scan_limit=3,
        source_backlog_target=10,
        hn_max_item_fetches_per_refresh=10,
    )
    engine = create_app_engine(settings.database_url)
    init_database(engine, settings.database_url)
    session_factory = create_session_factory(engine)
    fetched_items: list[int] = []

    async def fake_fetch_best_story_ids(client, limit: int) -> list[int]:
        assert limit == 3
        return [1, 2, 3]

    async def fake_fetch_item(client, item_id: int) -> dict:
        fetched_items.append(item_id)
        return hn_item(item_id)

    monkeypatch.setattr("app.pipeline.fetch_best_story_ids", fake_fetch_best_story_ids)
    monkeypatch.setattr("app.pipeline.fetch_item", fake_fetch_item)

    with session_factory() as session:
        assert insert_source_item(session, hn_item(1)) is not None

        inserted_count = asyncio.run(ingest_hacker_news(session, settings))

        assert inserted_count == 2
        assert fetched_items == [2, 3]
        source_item_ids = [
            source_item.source_item_id
            for source_item in session.scalars(select(SourceItem)).all()
        ]
        assert sorted(source_item_ids) == ["1", "2", "3"]


def test_ingest_hacker_news_skips_item_fetches_when_backlog_is_full(monkeypatch) -> None:
    settings = Settings(
        database_url="sqlite://",
        hn_story_scan_limit=3,
        source_backlog_target=1,
        hn_max_item_fetches_per_refresh=10,
    )
    engine = create_app_engine(settings.database_url)
    init_database(engine, settings.database_url)
    session_factory = create_session_factory(engine)
    id_fetch_limits: list[int] = []
    fetched_items: list[int] = []

    async def fake_fetch_best_story_ids(client, limit: int) -> list[int]:
        id_fetch_limits.append(limit)
        return [1, 2, 3]

    async def fake_fetch_item(client, item_id: int) -> dict:
        fetched_items.append(item_id)
        return hn_item(item_id)

    monkeypatch.setattr("app.pipeline.fetch_best_story_ids", fake_fetch_best_story_ids)
    monkeypatch.setattr("app.pipeline.fetch_item", fake_fetch_item)

    with session_factory() as session:
        assert insert_source_item(session, hn_item(1)) is not None

        inserted_count = asyncio.run(ingest_hacker_news(session, settings))

        assert inserted_count == 0
        assert id_fetch_limits == [3]
        assert fetched_items == []


def test_ingest_hacker_news_caps_item_fetches_per_refresh(monkeypatch) -> None:
    settings = Settings(
        database_url="sqlite://",
        hn_story_scan_limit=5,
        source_backlog_target=100,
        hn_max_item_fetches_per_refresh=2,
    )
    engine = create_app_engine(settings.database_url)
    init_database(engine, settings.database_url)
    session_factory = create_session_factory(engine)
    fetched_items: list[int] = []

    async def fake_fetch_best_story_ids(client, limit: int) -> list[int]:
        assert limit == 5
        return [1, 2, 3, 4, 5]

    async def fake_fetch_item(client, item_id: int) -> dict:
        fetched_items.append(item_id)
        return hn_item(item_id)

    monkeypatch.setattr("app.pipeline.fetch_best_story_ids", fake_fetch_best_story_ids)
    monkeypatch.setattr("app.pipeline.fetch_item", fake_fetch_item)

    with session_factory() as session:
        inserted_count = asyncio.run(ingest_hacker_news(session, settings))

        assert inserted_count == 2
        assert fetched_items == [1, 2]


def test_should_refresh_hacker_news_uses_startup_interval_and_low_backlog() -> None:
    settings = Settings(
        database_url="sqlite://",
        hn_refresh_interval_seconds=1800,
        source_backlog_low_watermark=10,
    )
    now = utc_now()

    assert should_refresh_hacker_news(
        now,
        None,
        100,
        settings,
        was_hn_backlog_below_low_watermark=False,
    )
    assert not should_refresh_hacker_news(
        now,
        now - timedelta(seconds=60),
        100,
        settings,
        was_hn_backlog_below_low_watermark=False,
    )
    assert should_refresh_hacker_news(
        now,
        now - timedelta(seconds=60),
        9,
        settings,
        was_hn_backlog_below_low_watermark=False,
    )
    assert not should_refresh_hacker_news(
        now,
        now - timedelta(seconds=60),
        9,
        settings,
        was_hn_backlog_below_low_watermark=True,
    )
    assert should_refresh_hacker_news(
        now,
        now - timedelta(seconds=1800),
        100,
        settings,
        was_hn_backlog_below_low_watermark=False,
    )


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
        post_generation_interval_seconds=60,
        post_generation_limit=1,
        image_output_dir=str(tmp_path / "images"),
        azure_openai_llm_endpoint="https://example.openai.azure.com",
        azure_openai_llm_api_key="llm-key",
        azure_openai_llm_deployment="llm",
        azure_openai_quality_filter_deployment="gpt-5.4-mini",
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

        async def evaluate_source_item_quality(
            self,
            source_item: SourceItem,
        ) -> SourceQualityEvaluation:
            return SourceQualityEvaluation(
                accepted=True,
                reason="Substantive source",
                categories=[],
            )

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


def test_generate_missing_posts_filters_rejected_source_item(tmp_path, monkeypatch) -> None:
    settings = Settings(
        database_url="sqlite://",
        post_generation_interval_seconds=60,
        post_generation_limit=1,
        image_output_dir=str(tmp_path / "images"),
        azure_openai_llm_endpoint="https://example.openai.azure.com",
        azure_openai_llm_api_key="llm-key",
        azure_openai_llm_deployment="llm",
        azure_openai_quality_filter_deployment="gpt-5.4-mini",
        azure_openai_image_endpoint="https://example.cognitiveservices.azure.com",
        azure_openai_image_api_key="image-key",
        azure_openai_image_deployment="image",
    )
    engine = create_app_engine(settings.database_url)
    init_database(engine, settings.database_url)
    session_factory = create_session_factory(engine)

    class RejectingAzureResponsesClient:
        def __init__(self, settings: Settings) -> None:
            pass

        async def evaluate_source_item_quality(
            self,
            source_item: SourceItem,
        ) -> SourceQualityEvaluation:
            return SourceQualityEvaluation(
                accepted=False,
                reason="Mostly promotional and thin.",
                categories=["advertisement", "low_information"],
            )

        async def generate_post_content(self, source_item: SourceItem) -> GeneratedPostContent:
            raise AssertionError("content generation should not run")

        async def generate_image(self, prompt: str) -> bytes:
            raise AssertionError("image generation should not run")

    async def fake_fetch_article_text(client, url: str) -> str:
        return f"Fetched article text from {url}"

    monkeypatch.setattr("app.pipeline.AzureResponsesClient", RejectingAzureResponsesClient)
    monkeypatch.setattr("app.pipeline.fetch_article_text", fake_fetch_article_text)

    with session_factory() as session:
        source_item = insert_source_item(session, hn_item(1))
        assert source_item is not None

        generated_count = asyncio.run(generate_missing_posts(session, settings))

        assert generated_count == 0
        posts = session.scalars(select(GeneratedPost)).all()
        assert len(posts) == 1
        assert posts[0].source_item_id == source_item.id
        assert posts[0].status == "filtered"
        assert posts[0].error_message is not None
        assert "Mostly promotional" in posts[0].error_message
        assert "advertisement, low_information" in posts[0].error_message
        assert source_items_without_generated_post(session, 10) == []


def test_generate_missing_posts_marks_article_fetch_failure_as_failed(
    tmp_path,
    monkeypatch,
) -> None:
    settings = Settings(
        database_url="sqlite://",
        post_generation_interval_seconds=60,
        post_generation_limit=1,
        image_output_dir=str(tmp_path / "images"),
        azure_openai_llm_endpoint="https://example.openai.azure.com",
        azure_openai_llm_api_key="llm-key",
        azure_openai_llm_deployment="llm",
        azure_openai_quality_filter_deployment="gpt-5.4-mini",
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
        post_generation_interval_seconds=60,
        image_output_dir=str(tmp_path / "images"),
        azure_openai_llm_endpoint="https://example.openai.azure.com",
        azure_openai_llm_api_key="llm-key",
        azure_openai_llm_deployment="llm",
        azure_openai_quality_filter_deployment="gpt-5.4-mini",
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

        async def evaluate_source_item_quality(
            self,
            source_item: SourceItem,
        ) -> SourceQualityEvaluation:
            return SourceQualityEvaluation(
                accepted=True,
                reason="Substantive source",
                categories=[],
            )

        async def generate_post_content(self, source_item: SourceItem) -> GeneratedPostContent:
            raise RuntimeError("temporary failure")

        async def generate_image(self, prompt: str) -> bytes:
            raise AssertionError("image generation should not run")

    class SuccessfulAzureResponsesClient:
        def __init__(self, settings: Settings) -> None:
            pass

        async def evaluate_source_item_quality(
            self,
            source_item: SourceItem,
        ) -> SourceQualityEvaluation:
            return SourceQualityEvaluation(
                accepted=True,
                reason="Substantive source",
                categories=[],
            )

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


def test_generate_missing_posts_does_not_refetch_empty_article_after_retry(
    tmp_path,
    monkeypatch,
) -> None:
    settings = Settings(
        database_url="sqlite://",
        post_generation_interval_seconds=60,
        image_output_dir=str(tmp_path / "images"),
        azure_openai_llm_endpoint="https://example.openai.azure.com",
        azure_openai_llm_api_key="llm-key",
        azure_openai_llm_deployment="llm",
        azure_openai_quality_filter_deployment="gpt-5.4-mini",
        azure_openai_image_endpoint="https://example.cognitiveservices.azure.com",
        azure_openai_image_api_key="image-key",
        azure_openai_image_deployment="image",
    )
    engine = create_app_engine(settings.database_url)
    init_database(engine, settings.database_url)
    session_factory = create_session_factory(engine)
    fetch_calls: list[str] = []
    content_article_texts: list[str | None] = []

    class FailingAzureResponsesClient:
        def __init__(self, settings: Settings) -> None:
            pass

        async def evaluate_source_item_quality(
            self,
            source_item: SourceItem,
        ) -> SourceQualityEvaluation:
            return SourceQualityEvaluation(
                accepted=True,
                reason="Substantive source",
                categories=[],
            )

        async def generate_post_content(self, source_item: SourceItem) -> GeneratedPostContent:
            content_article_texts.append(source_item.article_text)
            raise RuntimeError("temporary failure")

        async def generate_image(self, prompt: str) -> bytes:
            raise AssertionError("image generation should not run")

    class SuccessfulAzureResponsesClient:
        def __init__(self, settings: Settings) -> None:
            pass

        async def evaluate_source_item_quality(
            self,
            source_item: SourceItem,
        ) -> SourceQualityEvaluation:
            return SourceQualityEvaluation(
                accepted=True,
                reason="Substantive source",
                categories=[],
            )

        async def generate_post_content(self, source_item: SourceItem) -> GeneratedPostContent:
            content_article_texts.append(source_item.article_text)
            return generated_content()

        async def generate_image(self, prompt: str) -> bytes:
            return b"image-bytes"

    async def empty_fetch_article_text(client, url: str) -> str | None:
        fetch_calls.append(url)
        return None

    monkeypatch.setattr("app.pipeline.fetch_article_text", empty_fetch_article_text)

    with session_factory() as session:
        source_item = insert_source_item(session, hn_item(1))
        assert source_item is not None

        monkeypatch.setattr("app.pipeline.AzureResponsesClient", FailingAzureResponsesClient)
        failed_count = asyncio.run(generate_missing_posts(session, settings))

        assert failed_count == 0
        assert source_item.article_text is None
        assert source_item.article_fetched_at is not None
        assert fetch_calls == ["https://example.com/sqlite"]

        monkeypatch.setattr("app.pipeline.AzureResponsesClient", SuccessfulAzureResponsesClient)
        generated_count = asyncio.run(generate_missing_posts(session, settings))

        assert generated_count == 1
        assert fetch_calls == ["https://example.com/sqlite"]
        assert content_article_texts == [None, None]
        posts = session.scalars(select(GeneratedPost)).all()
        assert len(posts) == 1
        assert posts[0].status == "ready"
