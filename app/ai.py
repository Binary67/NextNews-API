import base64
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import quote

import httpx
from openai import AsyncOpenAI

from app.config import Settings
from app.models import ConversationMessage, GeneratedPost, SourceItem


@dataclass(frozen=True)
class GeneratedPostContent:
    title: str
    description: str
    content: str
    image_prompt: str


@dataclass(frozen=True)
class SourceQualityEvaluation:
    accepted: bool
    reason: str
    categories: list[str]


@dataclass(frozen=True)
class ThreadReplyCitation:
    url: str
    title: str | None = None


@dataclass(frozen=True)
class ThreadReply:
    content: str
    citations: list[ThreadReplyCitation]
    response_id: str | None = None


IMAGE_STYLES: tuple[str, ...] = (
    "Realistic editorial photograph with natural lighting, believable people or "
    "objects, and no text overlay.",
    "Documentary-style photograph with candid composition, natural setting, and "
    "no text overlay.",
    "Studio product photograph with clean lighting, object-focused composition, "
    "and no text overlay.",
    "Macro detail photograph emphasizing close-up technical texture, materials, "
    "or hardware details, with no text overlay.",
    "Polished editorial illustration with a modern magazine feel, clear subject, "
    "and no text overlay.",
    "Flat vector editorial illustration with clean geometric shapes, restrained "
    "colors, and no text overlay.",
    "Minimalist spot illustration with a simple focused visual metaphor and no "
    "text overlay.",
    "Isometric illustration showing structured digital systems, infrastructure, "
    "or workflows, with no text overlay.",
    "Technical cutaway illustration showing components, layers, or architecture "
    "without labels or text overlay.",
    "Conceptual editorial metaphor that represents the story idea symbolically, "
    "with no text overlay.",
    "Abstract data visual using grids, flows, particles, or network patterns, "
    "with no readable text.",
    "Soft 3D editorial render with polished objects, balanced lighting, and no "
    "text overlay.",
)


def random_image_style() -> str:
    return random.choice(IMAGE_STYLES)


def _response_value(item: object, key: str, default: object = None) -> object:
    if isinstance(item, dict):
        return item.get(key, default)

    return getattr(item, key, default)


def _extract_url_citations(response: object) -> list[ThreadReplyCitation]:
    citations: list[ThreadReplyCitation] = []
    seen_urls: set[str] = set()

    for output_item in _response_value(response, "output", []) or []:
        if _response_value(output_item, "type") != "message":
            continue

        for content_item in _response_value(output_item, "content", []) or []:
            for annotation in _response_value(content_item, "annotations", []) or []:
                if _response_value(annotation, "type") != "url_citation":
                    continue

                url = _response_value(annotation, "url")
                if not isinstance(url, str) or not url or url in seen_urls:
                    continue

                title = _response_value(annotation, "title")
                citations.append(
                    ThreadReplyCitation(
                        url=url,
                        title=title if isinstance(title, str) and title else None,
                    )
                )
                seen_urls.add(url)

    return citations


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
            "description": (
                "Prompt for a style-appropriate editorial image with no text overlay."
            ),
        },
    },
    "required": ["title", "description", "content", "image_prompt"],
}


QUALITY_FILTER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "accepted": {
            "type": "boolean",
            "description": "Whether the source item is worth turning into a NextNews post.",
        },
        "reason": {
            "type": "string",
            "description": "Brief explanation of the decision.",
        },
        "categories": {
            "type": "array",
            "description": "Rejection categories, or an empty array when accepted.",
            "items": {
                "type": "string",
                "enum": [
                    "clickbait",
                    "advertisement",
                    "flamebait",
                    "low_information",
                    "unsupported_by_source",
                    "off_topic",
                    "duplicate_or_meta",
                ],
            },
        },
    },
    "required": ["accepted", "reason", "categories"],
}


class AzureResponsesClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.azure_llm_configured:
            raise RuntimeError("Azure OpenAI LLM settings are not fully configured")

        self._settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.azure_openai_llm_endpoint,
            api_key=settings.azure_openai_llm_api_key_value,
        )

    async def evaluate_source_item_quality(
        self,
        source_item: SourceItem,
    ) -> SourceQualityEvaluation:
        response = await self._client.responses.create(
            model=self._settings.azure_openai_quality_filter_deployment,
            instructions=(
                "You are the quality gate for a technology news feed. Decide whether "
                "a source item is worth turning into a post. Accept only substantive, "
                "useful, factual items with enough source detail to support their main "
                "claim. Reject clickbait, advertisements, promotional content, ragebait, "
                "argument-bait, thin low-information content, off-topic content, mostly "
                "meta discussion, and claims not supported by the provided source text. "
                "Use only the provided source item; do not do external fact checking. "
                "Return structured JSON only."
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
                }
            ),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "nextnews_source_quality_filter",
                    "schema": QUALITY_FILTER_SCHEMA,
                    "strict": True,
                }
            },
        )

        payload = json.loads(response.output_text)
        return SourceQualityEvaluation(
            accepted=payload["accepted"],
            reason=payload["reason"].strip(),
            categories=list(payload["categories"]),
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
                    "image_style": random_image_style(),
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

    async def generate_thread_reply(
        self,
        post: GeneratedPost,
        prior_messages: Sequence[ConversationMessage],
        user_message: str,
    ) -> ThreadReply:
        source_item = post.source_item
        response = await self._client.responses.create(
            model=self._settings.azure_openai_llm_deployment,
            instructions=(
                "You answer user questions in a NextNews conversation thread. Ground "
                "the answer in the provided post context and same-thread conversation "
                "history. Use web search when current or external context would improve "
                "the answer. Do not use other post comments or other threads as context. "
                "Do not invent unsupported facts, numbers, quotes, or conclusions. Keep "
                "the answer clear, useful, and conversational."
            ),
            tools=[{"type": "web_search"}],
            tool_choice="auto",
            include=["web_search_call.action.sources"],
            input=json.dumps(
                {
                    "post": {
                        "title": post.title,
                        "description": post.description,
                        "content": post.content,
                        "source_url": source_item.url,
                        "article_text": source_item.article_text,
                    },
                    "thread_messages": [
                        {
                            "role": message.role,
                            "content": message.content,
                        }
                        for message in prior_messages
                    ],
                    "user_message": user_message,
                }
            ),
        )

        response_id = _response_value(response, "id")
        return ThreadReply(
            content=response.output_text.strip(),
            citations=_extract_url_citations(response),
            response_id=response_id if isinstance(response_id, str) else None,
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
