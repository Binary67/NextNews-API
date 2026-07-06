from fastapi.testclient import TestClient

from app.config import Settings
from app.database import create_app_engine, create_session_factory, init_database
from app.main import create_app
from app.models import GeneratedPost, SourceItem, utc_now


def make_client(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'nextnews-test.db'}",
        image_output_dir=str(tmp_path / "images"),
        hn_pipeline_interval_seconds=60,
    )
    app = create_app(settings=settings, start_pipeline=False)
    client = TestClient(app)
    return client, settings


def seed_posts(settings: Settings) -> tuple[int, int]:
    engine = create_app_engine(settings.database_url)
    init_database(engine, settings.database_url)
    session_factory = create_session_factory(engine)
    with session_factory() as session:
        source_item = SourceItem(
            source="hacker_news",
            source_item_id="1",
            title="Ready source",
            url="https://example.com/ready",
            author="author",
            score=10,
            raw_json={"id": 1},
        )
        hidden_source_item = SourceItem(
            source="hacker_news",
            source_item_id="2",
            title="Processing source",
            url="https://example.com/processing",
            author="author",
            score=5,
            raw_json={"id": 2},
        )
        session.add_all([source_item, hidden_source_item])
        session.commit()

        ready = GeneratedPost(
            source_item_id=source_item.id,
            title="Ready post",
            description="Ready description",
            content="Ready content",
            image_prompt="A realistic image",
            image_path=f"{settings.image_output_dir}/post-1.png",
            status="ready",
            ready_at=utc_now(),
        )
        processing = GeneratedPost(
            source_item_id=hidden_source_item.id,
            status="processing",
        )
        session.add_all([ready, processing])
        session.commit()
        return ready.id, processing.id


def test_posts_only_returns_ready_posts(tmp_path) -> None:
    client, settings = make_client(tmp_path)
    ready_id, _ = seed_posts(settings)

    with client:
        response = client.get("/posts")

    assert response.status_code == 200
    body = response.json()
    assert [post["id"] for post in body] == [ready_id]
    assert body[0]["image_url"] == "/images/post-1.png"


def test_post_detail_returns_ready_post_and_hides_non_ready(tmp_path) -> None:
    client, settings = make_client(tmp_path)
    ready_id, processing_id = seed_posts(settings)

    with client:
        ready_response = client.get(f"/posts/{ready_id}")
        processing_response = client.get(f"/posts/{processing_id}")
        missing_response = client.get("/posts/999")

    assert ready_response.status_code == 200
    assert ready_response.json()["title"] == "Ready post"
    assert processing_response.status_code == 404
    assert missing_response.status_code == 404
