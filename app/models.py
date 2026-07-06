from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.agents import DEFAULT_AGENT_NAME
from app.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SourceItem(Base):
    __tablename__ = "source_items"
    __table_args__ = (
        UniqueConstraint("source", "source_item_id", name="uq_source_items_source_item_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(64), index=True)
    source_item_id: Mapped[str] = mapped_column(String(128), index=True)
    title: Mapped[str] = mapped_column(String(500))
    url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    author: Mapped[str | None] = mapped_column(String(255), nullable=True)
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    article_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    article_fetched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    raw_json: Mapped[dict] = mapped_column(JSON)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    generated_post: Mapped["GeneratedPost | None"] = relationship(
        back_populates="source_item",
        uselist=False,
    )


class GeneratedPost(Base):
    __tablename__ = "generated_posts"
    __table_args__ = (
        UniqueConstraint("source_item_id", name="uq_generated_posts_source_item_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_item_id: Mapped[int] = mapped_column(ForeignKey("source_items.id"), index=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    agent_name: Mapped[str] = mapped_column(String(100), default=DEFAULT_AGENT_NAME)
    status: Mapped[str] = mapped_column(String(32), default="processing", index=True)
    llm_deployment: Mapped[str | None] = mapped_column(String(255), nullable=True)
    image_deployment: Mapped[str | None] = mapped_column(String(255), nullable=True)
    image_quality: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )
    ready_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    source_item: Mapped[SourceItem] = relationship(back_populates="generated_post")
    like: Mapped["PostLike | None"] = relationship(back_populates="post", uselist=False)
    conversation_threads: Mapped[list["ConversationThread"]] = relationship(
        back_populates="post",
    )


class PostLike(Base):
    __tablename__ = "post_likes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    post_id: Mapped[int] = mapped_column(
        ForeignKey("generated_posts.id"),
        unique=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    post: Mapped[GeneratedPost] = relationship(back_populates="like")


class ConversationThread(Base):
    __tablename__ = "conversation_threads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    post_id: Mapped[int] = mapped_column(ForeignKey("generated_posts.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    post: Mapped[GeneratedPost] = relationship(back_populates="conversation_threads")
    messages: Mapped[list["ConversationMessage"]] = relationship(
        back_populates="thread",
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    thread_id: Mapped[int] = mapped_column(
        ForeignKey("conversation_threads.id"),
        index=True,
    )
    role: Mapped[str] = mapped_column(String(20), index=True)
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    response_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    llm_deployment: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    thread: Mapped[ConversationThread] = relationship(back_populates="messages")
