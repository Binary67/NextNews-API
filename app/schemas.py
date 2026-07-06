from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    status: str


class PostListItem(BaseModel):
    id: int
    title: str
    description: str
    content: str
    image_url: str | None
    source_name: str
    source_item_id: str
    source_url: str | None
    created_at: datetime
    agent_name: str
    app_like_count: int
    liked_by_me: bool

    model_config = ConfigDict(from_attributes=True)


class PostDetail(PostListItem):
    pass


class PostInteractionState(BaseModel):
    post_id: int
    app_like_count: int
    liked_by_me: bool


class ConversationMessageRequest(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def message_must_not_be_empty(cls, value: str) -> str:
        message = value.strip()
        if not message:
            raise ValueError("Message must not be empty")
        return message


class ConversationCitation(BaseModel):
    url: str
    title: str | None = None


class ConversationMessageResponse(BaseModel):
    id: int
    thread_id: int
    role: str
    content: str
    citations: list[ConversationCitation] = Field(default_factory=list)
    response_id: str | None = None
    llm_deployment: str | None = None
    created_at: datetime


class ConversationThreadSummary(BaseModel):
    thread_id: int
    post_id: int
    message_count: int
    last_message: ConversationMessageResponse | None
    created_at: datetime
    updated_at: datetime


class ConversationThreadResponse(BaseModel):
    thread_id: int
    post_id: int
    created_at: datetime
    updated_at: datetime
    messages: list[ConversationMessageResponse]
