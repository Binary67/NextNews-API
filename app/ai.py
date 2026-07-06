import base64
import json
from dataclasses import dataclass
from urllib.parse import quote

import httpx
from openai import AsyncOpenAI

from app.config import Settings
from app.models import SourceItem


@dataclass(frozen=True)
class GeneratedPostContent:
    title: str
    description: str
    content: str
    image_prompt: str


POST_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {
            "type": "string",
            "description": "Attractive neutral social feed title.",
        },
        "description": {
            "type": "string",
            "description": "Natural social-media-style description.",
        },
        "content": {
            "type": "string",
            "description": "Informative social-media-style post based on the source material.",
        },
        "image_prompt": {
            "type": "string",
            "description": "Prompt for a realistic editorial image with no text overlay.",
        },
    },
    "required": ["title", "description", "content", "image_prompt"],
}


class AzureResponsesClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.azure_configured:
            raise RuntimeError("Azure OpenAI settings are not fully configured")

        self._settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.azure_openai_llm_endpoint,
            api_key=settings.azure_openai_llm_api_key_value,
        )

    async def generate_post_content(self, source_item: SourceItem) -> GeneratedPostContent:
        response = await self._client.responses.create(
            model=self._settings.azure_openai_llm_deployment,
            instructions=(
                "You turn source material into factual, social-media-style posts for a "
                "technology news feed. Use article_text as the primary source and metadata "
                "only for orientation. Do not mention Hacker News, the source platform, "
                "source site, source origin, author, score, or submission metadata in the "
                "title, description, or content unless it is central to the article itself. "
                "Write an attractive title, a natural description, and informative content. "
                "Do not invent unsupported facts, numbers, quotes, or conclusions. Keep the "
                "tone clear and informed. Return structured JSON only."
            ),
            input=json.dumps(
                {
                    "source": source_item.source,
                    "source_item_id": source_item.source_item_id,
                    "title": source_item.title,
                    "url": source_item.url,
                    "author": source_item.author,
                    "score": source_item.score,
                    "article_text": source_item.article_text,
                    "raw": source_item.raw_json,
                    "image_style": (
                        "Realistic editorial photograph, natural lighting, no illustration, "
                        "no cartoon style, no text overlay, no UI mockup unless the story is "
                        "specifically about software UI."
                    ),
                }
            ),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "nextnews_generated_post",
                    "schema": POST_SCHEMA,
                    "strict": True,
                }
            },
        )

        payload = json.loads(response.output_text)
        return GeneratedPostContent(
            title=payload["title"].strip(),
            description=payload["description"].strip(),
            content=payload["content"].strip(),
            image_prompt=payload["image_prompt"].strip(),
        )

    async def generate_image(self, prompt: str) -> bytes:
        endpoint = (self._settings.azure_openai_image_endpoint or "").rstrip("/")
        deployment = quote(self._settings.azure_openai_image_deployment or "", safe="")
        url = f"{endpoint}/openai/deployments/{deployment}/images/generations"
        body = {
            "prompt": prompt,
            "n": 1,
            "size": self._settings.image_size,
            "quality": self._settings.image_quality,
            "output_format": "png",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                url,
                headers={
                    "api-key": self._settings.azure_openai_image_api_key_value,
                    "Content-Type": "application/json",
                },
                params={"api-version": self._settings.azure_openai_image_api_version},
                json=body,
            )

        response_data = response.json()
        if response.is_error:
            error = response_data.get("error", {})
            code = error.get("code", "unknown_error")
            message = error.get("message", response.text)
            raise RuntimeError(f"Azure image generation failed: {code}: {message}")

        try:
            image_base64 = response_data["data"][0]["b64_json"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                "Azure image generation response did not include data[0].b64_json"
            ) from error

        if not image_base64:
            raise RuntimeError("Azure image generation response included empty b64_json")

        return base64.b64decode(image_base64)
