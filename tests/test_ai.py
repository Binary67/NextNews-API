import asyncio
import base64
import json

import pytest

from app.ai import POST_SCHEMA, QUALITY_FILTER_SCHEMA, AzureResponsesClient
from app.config import Settings
from app.models import SourceItem


class FakeImageResponse:
    def __init__(self, payload: dict, *, is_error: bool = False, text: str = "") -> None:
        self._payload = payload
        self.is_error = is_error
        self.text = text

    def json(self) -> dict:
        return self._payload


class FakeAsyncClient:
    requests: list[dict] = []
    response = FakeImageResponse({})

    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url, *, headers, params, json):
        self.requests.append(
            {
                "url": url,
                "headers": headers,
                "params": params,
                "json": json,
                "timeout": self.timeout,
            }
        )
        return self.response


class FakeResponses:
    requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        response_name = kwargs["text"]["format"]["name"]

        class Response:
            def __init__(self, output_text: str) -> None:
                self.output_text = output_text

        if response_name == "nextnews_source_quality_filter":
            return Response(
                json.dumps(
                    {
                        "accepted": False,
                        "reason": "The source is mostly promotional.",
                        "categories": ["advertisement"],
                    }
                )
            )

        return Response(
            json.dumps(
                {
                    "title": "Useful release for developers",
                    "description": "A new release improves developer workflows.",
                    "content": "The update focuses on practical improvements for teams.",
                    "image_prompt": "Realistic editorial photograph of software engineers",
                }
            )
        )


class FakeOpenAI:
    def __init__(self, *, base_url: str | None, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.responses = FakeResponses()


def configured_settings() -> Settings:
    return Settings(
        database_url="sqlite://",
        azure_openai_llm_endpoint="https://llm.example.com/openai/v1/",
        azure_openai_llm_api_key="llm-key",
        azure_openai_llm_deployment="llm-deployment",
        azure_openai_quality_filter_deployment="gpt-5.4-mini",
        azure_openai_image_endpoint="https://image.example.cognitiveservices.azure.com/",
        azure_openai_image_api_key="image-key",
        azure_openai_image_deployment="gpt-image-2-prod",
        azure_openai_image_api_version="2025-04-01-preview",
        image_quality="medium",
        image_size="1024x1024",
    )


def test_generate_post_content_uses_article_text_without_source_framing(
    monkeypatch,
) -> None:
    FakeResponses.requests = []
    monkeypatch.setattr("app.ai.AsyncOpenAI", FakeOpenAI)
    selected_image_style = (
        "Flat vector editorial illustration with clean geometric shapes, restrained "
        "colors, and no text overlay."
    )
    monkeypatch.setattr("app.ai.random_image_style", lambda: selected_image_style)

    source_item = SourceItem(
        source="hacker_news",
        source_item_id="123",
        title="Original submission title",
        url="https://example.com/article",
        author="author",
        score=42,
        article_text="The article explains why the release matters to developers.",
        raw_json={"id": 123},
    )

    client = AzureResponsesClient(configured_settings())
    result = asyncio.run(client.generate_post_content(source_item))

    assert result.title == "Useful release for developers"
    request = FakeResponses.requests[0]
    request_input = json.loads(request["input"])
    schema_text = json.dumps(POST_SCHEMA)

    assert request_input["article_text"] == source_item.article_text
    assert request_input["image_style"] == selected_image_style
    assert "Use article_text as the primary source" in request["instructions"]
    assert "Do not mention Hacker News" in request["instructions"]
    assert "realistic editorial image" not in schema_text
    assert "under " not in schema_text
    assert "concise" not in schema_text


def test_evaluate_source_item_quality_uses_filter_deployment(monkeypatch) -> None:
    FakeResponses.requests = []
    monkeypatch.setattr("app.ai.AsyncOpenAI", FakeOpenAI)
    source_item = SourceItem(
        source="hacker_news",
        source_item_id="123",
        title="Original submission title",
        url="https://example.com/article",
        author="author",
        score=42,
        article_text="The article is a thin product announcement.",
        raw_json={"id": 123},
    )

    client = AzureResponsesClient(configured_settings())
    result = asyncio.run(client.evaluate_source_item_quality(source_item))

    assert result.accepted is False
    assert result.categories == ["advertisement"]
    request = FakeResponses.requests[0]
    request_input = json.loads(request["input"])
    schema_text = json.dumps(QUALITY_FILTER_SCHEMA)

    assert request["model"] == "gpt-5.4-mini"
    assert request_input["article_text"] == source_item.article_text
    assert "quality gate" in request["instructions"]
    assert "unsupported_by_source" in schema_text


def test_generate_image_calls_azure_image_rest_endpoint(monkeypatch) -> None:
    image_bytes = b"png bytes"
    FakeAsyncClient.requests = []
    FakeAsyncClient.response = FakeImageResponse(
        {"data": [{"b64_json": base64.b64encode(image_bytes).decode("ascii")}]}
    )
    monkeypatch.setattr("app.ai.httpx.AsyncClient", FakeAsyncClient)

    client = AzureResponsesClient(configured_settings())
    result = asyncio.run(client.generate_image("realistic image prompt"))

    assert result == image_bytes
    assert FakeAsyncClient.requests == [
        {
            "url": (
                "https://image.example.cognitiveservices.azure.com"
                "/openai/deployments/gpt-image-2-prod/images/generations"
            ),
            "headers": {
                "api-key": "image-key",
                "Content-Type": "application/json",
            },
            "params": {"api-version": "2025-04-01-preview"},
            "json": {
                "prompt": "realistic image prompt",
                "n": 1,
                "size": "1024x1024",
                "quality": "medium",
                "output_format": "png",
            },
            "timeout": 120.0,
        }
    ]


def test_generate_image_raises_for_azure_error_payload(monkeypatch) -> None:
    FakeAsyncClient.requests = []
    FakeAsyncClient.response = FakeImageResponse(
        {"error": {"code": "contentFilter", "message": "blocked"}},
        is_error=True,
    )
    monkeypatch.setattr("app.ai.httpx.AsyncClient", FakeAsyncClient)

    client = AzureResponsesClient(configured_settings())

    with pytest.raises(RuntimeError, match="contentFilter: blocked"):
        asyncio.run(client.generate_image("blocked prompt"))


def test_generate_image_raises_for_missing_b64_json(monkeypatch) -> None:
    FakeAsyncClient.requests = []
    FakeAsyncClient.response = FakeImageResponse({"data": [{}]})
    monkeypatch.setattr("app.ai.httpx.AsyncClient", FakeAsyncClient)

    client = AzureResponsesClient(configured_settings())

    with pytest.raises(RuntimeError, match=r"data\[0\]\.b64_json"):
        asyncio.run(client.generate_image("prompt"))
