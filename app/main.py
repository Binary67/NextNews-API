import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, selectinload, sessionmaker

from app.ai import AzureResponsesClient, ThreadReply
from app.config import Settings, get_settings
from app.database import create_app_engine, create_session_factory, init_database
from app.models import (
    ConversationMessage,
    ConversationThread,
    GeneratedPost,
    PostLike,
    utc_now,
)
from app.pipeline import run_pipeline_loop
from app.schemas import (
    ConversationCitation,
    ConversationMessageRequest,
    ConversationMessageResponse,
    ConversationThreadResponse,
    ConversationThreadSummary,
    HealthResponse,
    PostDetail,
    PostInteractionState,
    PostListItem,
)


def _prune_old_logs(log_dir: Path, keep_count: int = 3) -> None:
    log_files = sorted(log_dir.glob("*.log"))
    for log_file in log_files[:-keep_count]:
        log_file.unlink()


def configure_app_logging() -> None:
    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.INFO)
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    has_file_handler = False
    for existing_handler in list(app_logger.handlers):
        if isinstance(existing_handler, logging.FileHandler):
            has_file_handler = True
            continue
        app_logger.removeHandler(existing_handler)
        existing_handler.close()

    if not has_file_handler:
        log_path = log_dir / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.log"
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        app_logger.addHandler(handler)
        _prune_old_logs(log_dir)
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


def _ready_thread_or_404(session: Session, thread_id: int) -> ConversationThread:
    thread = session.get(ConversationThread, thread_id)
    if thread is None or thread.post.status != "ready":
        raise HTTPException(status_code=404, detail="Thread not found")

    return thread


def _conversation_client_or_503(settings: Settings) -> AzureResponsesClient:
    if not settings.azure_llm_configured:
        raise HTTPException(status_code=503, detail="Azure OpenAI LLM is not configured")

    try:
        return AzureResponsesClient(settings)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def _message_to_schema(message: ConversationMessage) -> ConversationMessageResponse:
    return ConversationMessageResponse(
        id=message.id,
        thread_id=message.thread_id,
        role=message.role,
        content=message.content,
        citations=[
            ConversationCitation(
                url=citation["url"],
                title=citation.get("title"),
            )
            for citation in message.citations or []
            if citation.get("url")
        ],
        response_id=message.response_id,
        llm_deployment=message.llm_deployment,
        created_at=message.created_at,
    )


def _thread_messages(session: Session, thread_id: int) -> list[ConversationMessage]:
    statement = (
        select(ConversationMessage)
        .where(ConversationMessage.thread_id == thread_id)
        .order_by(ConversationMessage.created_at, ConversationMessage.id)
    )
    return list(session.scalars(statement).all())


def _thread_to_schema(
    thread: ConversationThread,
    messages: list[ConversationMessage],
) -> ConversationThreadResponse:
    return ConversationThreadResponse(
        thread_id=thread.id,
        post_id=thread.post_id,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
        messages=[_message_to_schema(message) for message in messages],
    )


def _thread_summary_to_schema(
    thread: ConversationThread,
    messages: list[ConversationMessage],
) -> ConversationThreadSummary:
    last_message = messages[-1] if messages else None
    return ConversationThreadSummary(
        thread_id=thread.id,
        post_id=thread.post_id,
        message_count=len(messages),
        last_message=_message_to_schema(last_message) if last_message is not None else None,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
    )


def _assistant_message(
    thread_id: int,
    reply: ThreadReply,
    settings: Settings,
) -> ConversationMessage:
    return ConversationMessage(
        thread_id=thread_id,
        role="assistant",
        content=reply.content,
        citations=[
            {
                "url": citation.url,
                "title": citation.title,
            }
            for citation in reply.citations
        ],
        response_id=reply.response_id,
        llm_deployment=settings.azure_openai_llm_deployment,
    )


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
            .options(
                selectinload(GeneratedPost.source_item),
                selectinload(GeneratedPost.like),
            )
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

    @app.post("/posts/{post_id}/threads", response_model=ConversationThreadResponse)
    async def create_thread(
        post_id: int,
        request: ConversationMessageRequest,
        session: Session = Depends(get_session),
    ) -> ConversationThreadResponse:
        post = _ready_post_or_404(session, post_id)
        ai_client = _conversation_client_or_503(app_settings)
        reply = await ai_client.generate_thread_reply(post, [], request.message)

        thread = ConversationThread(post_id=post.id)
        session.add(thread)
        session.flush()
        session.add_all(
            [
                ConversationMessage(
                    thread_id=thread.id,
                    role="user",
                    content=request.message,
                ),
                _assistant_message(thread.id, reply, app_settings),
            ]
        )
        session.commit()
        session.refresh(thread)

        return _thread_to_schema(thread, _thread_messages(session, thread.id))

    @app.get("/posts/{post_id}/threads", response_model=list[ConversationThreadSummary])
    def list_threads(
        post_id: int,
        session: Session = Depends(get_session),
    ) -> list[ConversationThreadSummary]:
        post = _ready_post_or_404(session, post_id)
        statement = (
            select(ConversationThread)
            .where(ConversationThread.post_id == post.id)
            .order_by(ConversationThread.updated_at.desc(), ConversationThread.id.desc())
        )
        threads = session.scalars(statement).all()
        return [
            _thread_summary_to_schema(thread, _thread_messages(session, thread.id))
            for thread in threads
        ]

    @app.get("/threads/{thread_id}", response_model=ConversationThreadResponse)
    def get_thread(
        thread_id: int,
        session: Session = Depends(get_session),
    ) -> ConversationThreadResponse:
        thread = _ready_thread_or_404(session, thread_id)
        return _thread_to_schema(thread, _thread_messages(session, thread.id))

    @app.post("/threads/{thread_id}/messages", response_model=ConversationThreadResponse)
    async def add_thread_message(
        thread_id: int,
        request: ConversationMessageRequest,
        session: Session = Depends(get_session),
    ) -> ConversationThreadResponse:
        thread = _ready_thread_or_404(session, thread_id)
        messages = _thread_messages(session, thread.id)
        ai_client = _conversation_client_or_503(app_settings)
        reply = await ai_client.generate_thread_reply(thread.post, messages, request.message)

        thread.updated_at = utc_now()
        session.add_all(
            [
                ConversationMessage(
                    thread_id=thread.id,
                    role="user",
                    content=request.message,
                ),
                _assistant_message(thread.id, reply, app_settings),
            ]
        )
        session.commit()
        session.refresh(thread)

        return _thread_to_schema(thread, _thread_messages(session, thread.id))

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
