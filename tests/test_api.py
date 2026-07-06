from fastapi.testclient import TestClient

from app.agents import DEFAULT_AGENT_NAME
from app.ai import ThreadReply, ThreadReplyCitation
from app.config import Settings
from app.database import create_app_engine, create_session_factory, init_database
from app.main import create_app
from app.models import GeneratedPost, SourceItem, utc_now


def make_client(tmp_path, *, azure_llm_configured: bool = False):
    settings_kwargs = {
        "database_url": f"sqlite:///{tmp_path / 'nextnews-test.db'}",
        "image_output_dir": str(tmp_path / "images"),
        "post_generation_interval_seconds": 60,
        "azure_openai_llm_endpoint": None,
        "azure_openai_llm_api_key": None,
        "azure_openai_llm_deployment": None,
    }
    if azure_llm_configured:
        settings_kwargs.update(
            {
                "azure_openai_llm_endpoint": "https://llm.example.com/openai/v1/",
                "azure_openai_llm_api_key": "llm-key",
                "azure_openai_llm_deployment": "llm-deployment",
            }
        )

    settings = Settings(**settings_kwargs)
    app = create_app(settings=settings, start_pipeline=False)
    client = TestClient(app)
    return client, settings


class FakeConversationClient:
    requests: list[dict] = []

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate_thread_reply(
        self,
        post: GeneratedPost,
        prior_messages,
        user_message: str,
    ) -> ThreadReply:
        self.requests.append(
            {
                "post_id": post.id,
                "prior_messages": [
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                    for message in prior_messages
                ],
                "user_message": user_message,
                "llm_deployment": self.settings.azure_openai_llm_deployment,
            }
        )
        reply_number = len(self.requests)
        return ThreadReply(
            content=f"Assistant reply {reply_number}",
            citations=[
                ThreadReplyCitation(
                    url=f"https://example.com/source-{reply_number}",
                    title=f"Source {reply_number}",
                )
            ],
            response_id=f"resp-{reply_number}",
        )


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
            article_text="Ready article text",
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


def test_create_thread_returns_user_and_assistant_messages(tmp_path, monkeypatch) -> None:
    FakeConversationClient.requests = []
    monkeypatch.setattr("app.main.AzureResponsesClient", FakeConversationClient)
    client, settings = make_client(tmp_path, azure_llm_configured=True)
    ready_id, _ = seed_posts(settings)

    with client:
        create_response = client.post(
            f"/posts/{ready_id}/threads",
            json={"message": " What does this mean for developers? "},
        )
        thread_id = create_response.json()["thread_id"]
        detail_response = client.get(f"/threads/{thread_id}")
        list_response = client.get(f"/posts/{ready_id}/threads")

    assert create_response.status_code == 200
    body = create_response.json()
    assert body["post_id"] == ready_id
    assert [message["role"] for message in body["messages"]] == ["user", "assistant"]
    assert body["messages"][0]["content"] == "What does this mean for developers?"
    assert body["messages"][1]["content"] == "Assistant reply 1"
    assert body["messages"][1]["citations"] == [
        {
            "url": "https://example.com/source-1",
            "title": "Source 1",
        }
    ]
    assert body["messages"][1]["response_id"] == "resp-1"
    assert body["messages"][1]["llm_deployment"] == "llm-deployment"
    assert FakeConversationClient.requests[0]["prior_messages"] == []
    assert FakeConversationClient.requests[0]["user_message"] == (
        "What does this mean for developers?"
    )
    assert FakeConversationClient.requests[0]["llm_deployment"] == "llm-deployment"
    assert detail_response.json() == body
    assert list_response.json()[0]["thread_id"] == thread_id
    assert list_response.json()[0]["message_count"] == 2
    assert list_response.json()[0]["last_message"]["content"] == "Assistant reply 1"


def test_continuing_thread_uses_only_that_thread_history(tmp_path, monkeypatch) -> None:
    FakeConversationClient.requests = []
    monkeypatch.setattr("app.main.AzureResponsesClient", FakeConversationClient)
    client, settings = make_client(tmp_path, azure_llm_configured=True)
    ready_id, _ = seed_posts(settings)

    with client:
        first_thread = client.post(
            f"/posts/{ready_id}/threads",
            json={"message": "First thread question"},
        ).json()
        second_thread = client.post(
            f"/posts/{ready_id}/threads",
            json={"message": "Second thread question"},
        ).json()
        continued = client.post(
            f"/threads/{first_thread['thread_id']}/messages",
            json={"message": "Follow up"},
        )
        second_detail = client.get(f"/threads/{second_thread['thread_id']}")

    assert continued.status_code == 200
    assert [message["content"] for message in continued.json()["messages"]] == [
        "First thread question",
        "Assistant reply 1",
        "Follow up",
        "Assistant reply 3",
    ]
    assert FakeConversationClient.requests[2]["prior_messages"] == [
        {"role": "user", "content": "First thread question"},
        {"role": "assistant", "content": "Assistant reply 1"},
    ]
    assert "Second thread question" not in [
        message["content"] for message in continued.json()["messages"]
    ]
    assert [message["content"] for message in second_detail.json()["messages"]] == [
        "Second thread question",
        "Assistant reply 2",
    ]


def test_thread_endpoints_hide_missing_and_non_ready_posts(tmp_path, monkeypatch) -> None:
    FakeConversationClient.requests = []
    monkeypatch.setattr("app.main.AzureResponsesClient", FakeConversationClient)
    client, settings = make_client(tmp_path, azure_llm_configured=True)
    _, processing_id = seed_posts(settings)

    with client:
        non_ready_create = client.post(
            f"/posts/{processing_id}/threads",
            json={"message": "Question"},
        )
        missing_create = client.post("/posts/999/threads", json={"message": "Question"})
        non_ready_list = client.get(f"/posts/{processing_id}/threads")
        missing_thread = client.get("/threads/999")
        missing_thread_message = client.post(
            "/threads/999/messages",
            json={"message": "Question"},
        )

    assert non_ready_create.status_code == 404
    assert missing_create.status_code == 404
    assert non_ready_list.status_code == 404
    assert missing_thread.status_code == 404
    assert missing_thread_message.status_code == 404


def test_thread_message_request_rejects_empty_messages(tmp_path, monkeypatch) -> None:
    FakeConversationClient.requests = []
    monkeypatch.setattr("app.main.AzureResponsesClient", FakeConversationClient)
    client, settings = make_client(tmp_path, azure_llm_configured=True)
    ready_id, _ = seed_posts(settings)

    with client:
        create_empty = client.post(
            f"/posts/{ready_id}/threads",
            json={"message": "   "},
        )
        thread = client.post(
            f"/posts/{ready_id}/threads",
            json={"message": "Question"},
        ).json()
        continue_empty = client.post(
            f"/threads/{thread['thread_id']}/messages",
            json={"message": "\n\t"},
        )

    assert create_empty.status_code == 422
    assert continue_empty.status_code == 422


def test_thread_creation_requires_azure_llm_config(tmp_path, monkeypatch) -> None:
    FakeConversationClient.requests = []
    monkeypatch.setattr("app.main.AzureResponsesClient", FakeConversationClient)
    client, settings = make_client(tmp_path)
    ready_id, _ = seed_posts(settings)

    with client:
        response = client.post(
            f"/posts/{ready_id}/threads",
            json={"message": "Question"},
        )

    assert response.status_code == 503
    assert FakeConversationClient.requests == []


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
