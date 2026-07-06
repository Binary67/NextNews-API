from fastapi.testclient import TestClient

from app.agents import DEFAULT_AGENT_NAME
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
            agent_name=DEFAULT_AGENT_NAME,
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
    assert body[0]["agent_name"] == DEFAULT_AGENT_NAME
    assert body[0]["app_like_count"] == 0
    assert body[0]["liked_by_me"] is False


def test_post_detail_returns_ready_post_and_hides_non_ready(tmp_path) -> None:
    client, settings = make_client(tmp_path)
    ready_id, processing_id = seed_posts(settings)

    with client:
        ready_response = client.get(f"/posts/{ready_id}")
        processing_response = client.get(f"/posts/{processing_id}")
        missing_response = client.get("/posts/999")

    assert ready_response.status_code == 200
    ready_body = ready_response.json()
    assert ready_body["title"] == "Ready post"
    assert ready_body["agent_name"] == DEFAULT_AGENT_NAME
    assert ready_body["app_like_count"] == 0
    assert ready_body["liked_by_me"] is False
    assert processing_response.status_code == 404
    assert missing_response.status_code == 404


def test_like_and_unlike_post_are_idempotent(tmp_path) -> None:
    client, settings = make_client(tmp_path)
    ready_id, _ = seed_posts(settings)

    with client:
        first_like = client.post(f"/posts/{ready_id}/like")
        second_like = client.post(f"/posts/{ready_id}/like")
        liked_detail = client.get(f"/posts/{ready_id}")
        first_unlike = client.delete(f"/posts/{ready_id}/like")
        second_unlike = client.delete(f"/posts/{ready_id}/like")
        unliked_detail = client.get(f"/posts/{ready_id}")

    liked_state = {
        "post_id": ready_id,
        "app_like_count": 1,
        "liked_by_me": True,
    }
    unliked_state = {
        "post_id": ready_id,
        "app_like_count": 0,
        "liked_by_me": False,
    }

    assert first_like.status_code == 200
    assert first_like.json() == liked_state
    assert second_like.status_code == 200
    assert second_like.json() == liked_state
    assert liked_detail.json()["app_like_count"] == 1
    assert liked_detail.json()["liked_by_me"] is True

    assert first_unlike.status_code == 200
    assert first_unlike.json() == unliked_state
    assert second_unlike.status_code == 200
    assert second_unlike.json() == unliked_state
    assert unliked_detail.json()["app_like_count"] == 0
    assert unliked_detail.json()["liked_by_me"] is False


def test_like_endpoints_hide_missing_and_non_ready_posts(tmp_path) -> None:
    client, settings = make_client(tmp_path)
    _, processing_id = seed_posts(settings)

    with client:
        processing_like = client.post(f"/posts/{processing_id}/like")
        missing_like = client.post("/posts/999/like")
        processing_unlike = client.delete(f"/posts/{processing_id}/like")
        missing_unlike = client.delete("/posts/999/like")

    assert processing_like.status_code == 404
    assert missing_like.status_code == 404
    assert processing_unlike.status_code == 404
    assert missing_unlike.status_code == 404
