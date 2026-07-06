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

    model_config = ConfigDict(from_attributes=True)


class PostDetail(PostListItem):
    pass

