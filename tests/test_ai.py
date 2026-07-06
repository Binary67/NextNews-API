import asyncio
import base64

import pytest

from app.ai import AzureResponsesClient
from app.config import Settings


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


def configured_settings() -> Settings:
    return Settings(
        database_url="sqlite://",
        azure_openai_llm_endpoint="https://llm.example.com/openai/v1/",
        azure_openai_llm_api_key="llm-key",
        azure_openai_llm_deployment="llm-deployment",
        azure_openai_image_endpoint="https://image.example.cognitiveservices.azure.com/",
        azure_openai_image_api_key="image-key",
        azure_openai_image_deployment="gpt-image-2-prod",
        azure_openai_image_api_version="2025-04-01-preview",
        image_quality="medium",
        image_size="1024x1024",
    )


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

