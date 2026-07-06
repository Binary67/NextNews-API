from datetime import datetime

from pydantic import BaseModel, ConfigDict


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
