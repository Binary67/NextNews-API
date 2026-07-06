import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import Select, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

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


def source_items_without_generated_post(session: Session, limit: int) -> list[SourceItem]:
    statement: Select[tuple[SourceItem]] = (
        select(SourceItem)
        .outerjoin(GeneratedPost, GeneratedPost.source_item_id == SourceItem.id)
        .where(GeneratedPost.id.is_(None))
        .order_by(SourceItem.fetched_at.desc(), SourceItem.id.desc())
        .limit(limit)
    )
    return list(session.scalars(statement))


async def ingest_hacker_news(session: Session, settings: Settings) -> int:
    inserted_count = 0
    async with httpx.AsyncClient(timeout=15.0) as client:
        logger.info("Fetching up to %s Hacker News story ids", settings.hn_fetch_limit)
        story_ids = await fetch_best_story_ids(client, settings.hn_fetch_limit)
        logger.info("Fetched %s Hacker News story ids", len(story_ids))
        for story_id in story_ids:
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

    for source_item in source_items:
        try:
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
) -> None:
    logger.info("Pipeline pass started")
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

        logger.info(
            "Sleeping %s seconds before next pipeline pass",
            settings.hn_pipeline_interval_seconds,
        )
        await asyncio.sleep(settings.hn_pipeline_interval_seconds)
