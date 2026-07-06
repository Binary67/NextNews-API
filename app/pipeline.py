import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import Select, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.article import extract_html_snippet_text, fetch_article_text
from app.agents import random_agent_name
from app.ai import AzureResponsesClient, GeneratedPostContent
from app.config import Settings
from app.hn import HN_SOURCE, fetch_best_story_ids, fetch_item, is_valid_story
from app.models import GeneratedPost, SourceItem, utc_now


logger = logging.getLogger(__name__)


def insert_source_item(session: Session, item: dict[str, Any]) -> SourceItem | None:
    source_item = SourceItem(
        source=HN_SOURCE,
        source_item_id=str(item["id"]),
        title=item["title"],
        url=item.get("url"),
        author=item.get("by"),
        score=item.get("score"),
        article_text=extract_html_snippet_text(item["text"]) if item.get("text") else None,
        raw_json=item,
    )
    session.add(source_item)

    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        return None

    session.refresh(source_item)
    return source_item


def create_ready_generated_post(
    session: Session,
    source_item: SourceItem,
    settings: Settings,
    content: GeneratedPostContent,
    image_bytes: bytes,
    output_dir: Path,
) -> GeneratedPost | None:
    post = GeneratedPost(
        source_item_id=source_item.id,
        title=content.title,
        description=content.description,
        content=content.content,
        image_prompt=content.image_prompt,
        agent_name=random_agent_name(),
        status="ready",
        llm_deployment=settings.azure_openai_llm_deployment,
        image_deployment=settings.azure_openai_image_deployment,
        image_quality=settings.image_quality,
        ready_at=utc_now(),
    )
    session.add(post)

    try:
        session.flush()
        image_path = output_dir / f"post-{post.id}.png"
        image_path.write_bytes(image_bytes)
        post.image_path = str(image_path)
        session.commit()
    except IntegrityError:
        session.rollback()
        return None
    except Exception:
        session.rollback()
        raise

    session.refresh(post)
    return post


def create_failed_generated_post(
    session: Session,
    source_item: SourceItem,
    error_message: str,
) -> GeneratedPost | None:
    post = GeneratedPost(
        source_item_id=source_item.id,
        status="failed",
        error_message=error_message,
    )
    session.add(post)

    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        return None

    session.refresh(post)
    return post


def article_fetch_error_message(error: Exception) -> str:
    message = str(error).strip() or error.__class__.__name__
    return f"Article fetch failed: {message}"[:1000]


def source_items_without_generated_post(session: Session, limit: int) -> list[SourceItem]:
    statement: Select[tuple[SourceItem]] = (
        select(SourceItem)
        .outerjoin(GeneratedPost, GeneratedPost.source_item_id == SourceItem.id)
        .where(GeneratedPost.id.is_(None))
        .order_by(SourceItem.fetched_at.asc(), SourceItem.id.asc())
        .limit(limit)
    )
    return list(session.scalars(statement))


def count_source_items_without_generated_post(
    session: Session,
    source: str | None = None,
) -> int:
    statement = (
        select(func.count(SourceItem.id))
        .outerjoin(GeneratedPost, GeneratedPost.source_item_id == SourceItem.id)
        .where(GeneratedPost.id.is_(None))
    )
    if source is not None:
        statement = statement.where(SourceItem.source == source)

    return int(session.scalar(statement) or 0)


def existing_source_item_ids(
    session: Session,
    source: str,
    source_item_ids: list[str],
) -> set[str]:
    if not source_item_ids:
        return set()

    statement = select(SourceItem.source_item_id).where(
        SourceItem.source == source,
        SourceItem.source_item_id.in_(source_item_ids),
    )
    return set(session.scalars(statement))


async def ensure_article_text(
    session: Session,
    client: httpx.AsyncClient,
    source_item: SourceItem,
) -> None:
    if source_item.article_text or not source_item.url:
        return

    source_item.article_text = await fetch_article_text(client, source_item.url)
    session.commit()


async def ingest_hacker_news(session: Session, settings: Settings) -> int:
    inserted_count = 0
    async with httpx.AsyncClient(timeout=15.0) as client:
        logger.info("Fetching up to %s Hacker News story ids", settings.hn_story_scan_limit)
        story_ids = await fetch_best_story_ids(client, settings.hn_story_scan_limit)
        logger.info("Fetched %s Hacker News story ids", len(story_ids))
        story_id_strings = [str(story_id) for story_id in story_ids]
        existing_story_ids = existing_source_item_ids(session, HN_SOURCE, story_id_strings)
        hn_backlog_count = count_source_items_without_generated_post(session, HN_SOURCE)
        item_fetch_limit = min(
            max(settings.source_backlog_target - hn_backlog_count, 0),
            settings.hn_max_item_fetches_per_refresh,
        )
        if item_fetch_limit == 0:
            logger.info(
                "Hacker News backlog has %s source items; skipping item fetches",
                hn_backlog_count,
            )
            return inserted_count

        unseen_story_ids = [
            story_id for story_id in story_ids if str(story_id) not in existing_story_ids
        ]
        logger.info(
            "Found %s unseen Hacker News ids; fetching up to %s items",
            len(unseen_story_ids),
            item_fetch_limit,
        )
        for story_id in unseen_story_ids[:item_fetch_limit]:
            logger.info("Fetching Hacker News item %s", story_id)
            item = await fetch_item(client, story_id)
            if not is_valid_story(item):
                logger.info("Skipping invalid Hacker News item %s", story_id)
                continue

            inserted = insert_source_item(session, item)
            if inserted is not None:
                inserted_count += 1
                logger.info(
                    "Inserted source item %s for Hacker News item %s",
                    inserted.id,
                    story_id,
                )
            else:
                logger.info("Hacker News item %s already exists; skipping insert", story_id)

    return inserted_count


def should_refresh_hacker_news(
    now: datetime,
    last_hn_refresh_attempt_at: datetime | None,
    hn_backlog_count: int,
    settings: Settings,
    *,
    was_hn_backlog_below_low_watermark: bool,
) -> bool:
    if last_hn_refresh_attempt_at is None:
        return True

    seconds_since_refresh_attempt = (now - last_hn_refresh_attempt_at).total_seconds()
    if seconds_since_refresh_attempt >= settings.hn_refresh_interval_seconds:
        return True

    return (
        hn_backlog_count < settings.source_backlog_low_watermark
        and not was_hn_backlog_below_low_watermark
    )


async def generate_missing_posts(session: Session, settings: Settings) -> int:
    if not settings.azure_configured:
        logger.info("Azure OpenAI is not configured; skipping post generation")
        return 0

    ai_client = AzureResponsesClient(settings)
    generated_count = 0
    output_dir = Path(settings.image_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_items = source_items_without_generated_post(session, settings.post_generation_limit)
    logger.info(
        "Found %s source items without generated posts (generation limit=%s)",
        len(source_items),
        settings.post_generation_limit,
    )

    async with httpx.AsyncClient(timeout=30.0) as article_client:
        for source_item in source_items:
            try:
                try:
                    await ensure_article_text(session, article_client, source_item)
                except Exception as error:
                    logger.exception(
                        "Failed to fetch article text for source item %s",
                        source_item.id,
                    )
                    session.rollback()
                    failed_post = create_failed_generated_post(
                        session,
                        source_item,
                        article_fetch_error_message(error),
                    )
                    if failed_post is None:
                        logger.info(
                            "Source item %s already has a generated post; skipping",
                            source_item.id,
                        )
                    continue

                logger.info(
                    "Generating post content from source item %s",
                    source_item.id,
                )
                content = await ai_client.generate_post_content(source_item)
                logger.info("Generating image for source item %s", source_item.id)
                image_bytes = await ai_client.generate_image(content.image_prompt)
                post = create_ready_generated_post(
                    session,
                    source_item,
                    settings,
                    content,
                    image_bytes,
                    output_dir,
                )
                if post is None:
                    logger.info(
                        "Source item %s already has a generated post; skipping",
                        source_item.id,
                    )
                    continue
                generated_count += 1
                logger.info("Generated post %s successfully", post.id)
            except Exception:
                logger.exception("Failed to generate post for source item %s", source_item.id)
                session.rollback()

    return generated_count


async def run_pipeline_once(
    session_factory: sessionmaker[Session],
    settings: Settings,
    *,
    refresh_hacker_news: bool = True,
) -> None:
    logger.info("Pipeline pass started")
    with session_factory() as session:
        inserted_count = 0
        if refresh_hacker_news:
            inserted_count = await ingest_hacker_news(session, settings)
        generated_count = await generate_missing_posts(session, settings)

    logger.info(
        "Pipeline pass complete: inserted_source_items=%s generated_posts=%s",
        inserted_count,
        generated_count,
    )


async def run_pipeline_loop(
    session_factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    last_hn_refresh_attempt_at: datetime | None = None
    was_hn_backlog_below_low_watermark = False
    while True:
        try:
            now = utc_now()
            with session_factory() as session:
                hn_backlog_count = count_source_items_without_generated_post(session, HN_SOURCE)
            refresh_hacker_news = should_refresh_hacker_news(
                now,
                last_hn_refresh_attempt_at,
                hn_backlog_count,
                settings,
                was_hn_backlog_below_low_watermark=was_hn_backlog_below_low_watermark,
            )
            was_hn_backlog_below_low_watermark = (
                hn_backlog_count < settings.source_backlog_low_watermark
            )
            if refresh_hacker_news:
                last_hn_refresh_attempt_at = now

            await run_pipeline_once(
                session_factory,
                settings,
                refresh_hacker_news=refresh_hacker_news,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Pipeline pass failed")

        logger.info(
            "Sleeping %s seconds before next pipeline pass",
            settings.post_generation_interval_seconds,
        )
        await asyncio.sleep(settings.post_generation_interval_seconds)
