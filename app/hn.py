from typing import Any

import httpx


HN_BASE_URL = "https://hacker-news.firebaseio.com/v0"
HN_SOURCE = "hacker_news"


def is_valid_story(item: dict[str, Any] | None) -> bool:
    if not item:
        return False

    return (
        item.get("type") == "story"
        and not item.get("deleted", False)
        and not item.get("dead", False)
        and bool(item.get("id"))
        and bool(item.get("title"))
    )


async def fetch_best_story_ids(client: httpx.AsyncClient, limit: int) -> list[int]:
    response = await client.get(f"{HN_BASE_URL}/beststories.json")
    response.raise_for_status()
    story_ids = response.json()
    return [int(story_id) for story_id in story_ids[:limit]]


async def fetch_item(client: httpx.AsyncClient, item_id: int) -> dict[str, Any] | None:
    response = await client.get(f"{HN_BASE_URL}/item/{item_id}.json")
    response.raise_for_status()
    return response.json()

