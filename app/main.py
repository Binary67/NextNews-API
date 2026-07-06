import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.database import create_app_engine, create_session_factory, init_database
from app.models import GeneratedPost, PostLike, SourceItem
from app.pipeline import run_pipeline_loop
from app.schemas import HealthResponse, PostDetail, PostInteractionState, PostListItem


def configure_app_logging() -> None:
    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.INFO)
    if not app_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        app_logger.addHandler(handler)
    app_logger.propagate = False


def _image_url(image_path: str | None, settings: Settings) -> str | None:
    if not image_path:
        return None

    image_name = Path(image_path).name
    return f"/images/{image_name}"


def _post_to_schema(post: GeneratedPost, settings: Settings) -> PostListItem:
    source_item = post.source_item
    liked_by_me = post.like is not None
    return PostListItem(
        id=post.id,
        title=post.title or "",
        description=post.description or "",
        content=post.content or "",
        image_url=_image_url(post.image_path, settings),
        source_name=source_item.source,
        source_item_id=source_item.source_item_id,
        source_url=source_item.url,
        created_at=post.ready_at or post.created_at,
        agent_name=post.agent_name,
        app_like_count=1 if liked_by_me else 0,
        liked_by_me=liked_by_me,
    )


def _post_interaction_state(post_id: int, liked_by_me: bool) -> PostInteractionState:
    return PostInteractionState(
        post_id=post_id,
        app_like_count=1 if liked_by_me else 0,
        liked_by_me=liked_by_me,
    )


def _ready_post_or_404(session: Session, post_id: int) -> GeneratedPost:
    post = session.get(GeneratedPost, post_id)
    if post is None or post.status != "ready":
        raise HTTPException(status_code=404, detail="Post not found")

    return post


def create_app(
    settings: Settings | None = None,
    *,
    start_pipeline: bool = True,
) -> FastAPI:
    configure_app_logging()
    app_settings = settings or get_settings()
    engine: Engine = create_app_engine(app_settings.database_url)
    session_factory: sessionmaker[Session] = create_session_factory(engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        init_database(engine, app_settings.database_url)
        Path(app_settings.image_output_dir).mkdir(parents=True, exist_ok=True)

        pipeline_task: asyncio.Task[None] | None = None
        if start_pipeline:
            pipeline_task = asyncio.create_task(run_pipeline_loop(session_factory, app_settings))

        yield

        if pipeline_task is not None:
            pipeline_task.cancel()
            try:
                await pipeline_task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title="NextNews API", version="0.1.0", lifespan=lifespan)
    app.mount(
        "/images",
        StaticFiles(directory=app_settings.image_output_dir, check_dir=False),
        name="images",
    )

    def get_session() -> Session:
        with session_factory() as session:
            yield session

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/posts", response_model=list[PostListItem])
    def list_posts(
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        session: Session = Depends(get_session),
    ) -> list[PostListItem]:
        statement = (
            select(GeneratedPost)
            .join(SourceItem)
            .where(GeneratedPost.status == "ready")
            .order_by(GeneratedPost.ready_at.desc(), GeneratedPost.id.desc())
            .offset(offset)
            .limit(limit)
        )
        posts = session.scalars(statement).all()
        return [_post_to_schema(post, app_settings) for post in posts]

    @app.get("/posts/{post_id}", response_model=PostDetail)
    def get_post(
        post_id: int,
        session: Session = Depends(get_session),
    ) -> PostDetail:
        post = _ready_post_or_404(session, post_id)

        return PostDetail(**_post_to_schema(post, app_settings).model_dump())

    @app.post("/posts/{post_id}/like", response_model=PostInteractionState)
    def like_post(
        post_id: int,
        session: Session = Depends(get_session),
    ) -> PostInteractionState:
        post = _ready_post_or_404(session, post_id)
        if post.like is None:
            session.add(PostLike(post_id=post.id))
            try:
                session.commit()
            except IntegrityError:
                session.rollback()

        return _post_interaction_state(post.id, liked_by_me=True)

    @app.delete("/posts/{post_id}/like", response_model=PostInteractionState)
    def unlike_post(
        post_id: int,
        session: Session = Depends(get_session),
    ) -> PostInteractionState:
        post = _ready_post_or_404(session, post_id)
        like = session.scalar(select(PostLike).where(PostLike.post_id == post.id))
        if like is not None:
            session.delete(like)
            session.commit()

        return _post_interaction_state(post.id, liked_by_me=False)

    return app


app = create_app()
