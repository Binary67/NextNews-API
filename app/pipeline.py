import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import Select, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.ai import AzureResponsesClient
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


def claim_generated_post(
    session: Session,
    source_item: SourceItem,
    settings: Settings,
) -> GeneratedPost | None:
    post = GeneratedPost(
        source_item_id=source_item.id,
        status="processing",
        llm_deployment=settings.azure_openai_llm_deployment,
        image_deployment=settings.azure_openai_image_deployment,
        image_quality=settings.image_quality,
    )
    session.add(post)

    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        return None

    session.refresh(post)
    return post


def source_items_without_generated_post(session: Session, limit: int) -> list[SourceItem]:
    statement: Select[tuple[SourceItem]] = (
        select(SourceItem)
        .outerjoin(GeneratedPost, GeneratedPost.source_item_id == SourceItem.id)
        .where(GeneratedPost.id.is_(None))
        .order_by(SourceItem.fetched_at.desc(), SourceItem.id.desc())
        .limit(limit)
    )
    return list(session.scalars(statement))


def mark_post_ready(
    session: Session,
    post: GeneratedPost,
    content_title: str,
    description: str,
    content: str,
    image_prompt: str,
    image_path: str,
) -> None:
    post.title = content_title
    post.description = description
    post.content = content
    post.image_prompt = image_prompt
    post.image_path = image_path
    post.status = "ready"
    post.error_message = None
    post.ready_at = utc_now()
    session.commit()


def mark_post_failed(session: Session, post: GeneratedPost, error: Exception) -> None:
    post.status = "failed"
    post.error_message = str(error)
    session.commit()


async def ingest_hacker_news(session: Session, settings: Settings) -> int:
    inserted_count = 0
    async with httpx.AsyncClient(timeout=15.0) as client:
        story_ids = await fetch_best_story_ids(client, settings.hn_fetch_limit)
        for story_id in story_ids:
            item = await fetch_item(client, story_id)
            if not is_valid_story(item):
                continue

            inserted = insert_source_item(session, item)
            if inserted is not None:
                inserted_count += 1

    return inserted_count


async def generate_missing_posts(session: Session, settings: Settings) -> int:
    if not settings.azure_configured:
        logger.info("Azure OpenAI is not configured; skipping post generation")
        return 0

    ai_client = AzureResponsesClient(settings)
    generated_count = 0
    output_dir = Path(settings.image_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for source_item in source_items_without_generated_post(session, settings.post_generation_limit):
        post = claim_generated_post(session, source_item, settings)
        if post is None:
            continue

        try:
            content = await ai_client.generate_post_content(source_item)
            image_bytes = await ai_client.generate_image(content.image_prompt)
            image_path = output_dir / f"post-{post.id}.png"
            image_path.write_bytes(image_bytes)
            mark_post_ready(
                session,
                post,
                content.title,
                content.description,
                content.content,
                content.image_prompt,
                str(image_path),
            )
            generated_count += 1
        except Exception as error:
            logger.exception("Failed to generate post for source item %s", source_item.id)
            mark_post_failed(session, post, error)

    return generated_count


async def run_pipeline_once(
    session_factory: sessionmaker[Session],
    settings: Settings,
) -> None:
    with session_factory() as session:
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
    while True:
        try:
            await run_pipeline_once(session_factory, settings)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Pipeline pass failed")

        await asyncio.sleep(settings.hn_pipeline_interval_seconds)
