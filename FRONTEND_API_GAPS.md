# Frontend API Gaps

## Current Status

The iOS frontend can already render real posts from the existing API:

- `GET /posts?limit=20&offset=0`
- `GET /posts/{post_id}`
- Static images under `/images/{filename}`

Current response shape:

```json
{
  "id": 4,
  "title": "HN spotlight: Starring the Computer",
  "description": "A short summary.",
  "content": "Longer generated post body.",
  "image_url": "/images/post-4.png",
  "source_name": "hacker_news",
  "source_item_id": "48796093",
  "source_url": "https://example.com/article",
  "created_at": "2026-07-06T07:52:25.654423"
}
```

This is enough for a basic feed. The next backend work should focus only on fields the app actually needs now.

## Updated Scope

### Categories Are Out Of Scope For Now

Do not add `category`, `primary_topic`, or `topics` yet.

The current app should remove category pills and show a single latest-posts feed. Categories can come back later when there is enough content volume and a clearer product need.

When categories return, avoid starting with a narrow hard-coded enum unless the product intentionally becomes an editorial sectioned feed.

### Hacker News Engagement Is Not Frontend Engagement

Do not add Hacker News score or Hacker News comment count to the frontend response for now.

These source metrics can stay internal for ranking, selection, or generation decisions, but they should not be displayed as NextNews likes or comments.

Avoid these frontend fields for now:

```json
{
  "score": 212,
  "comment_count": 47,
  "source_score": 212,
  "source_comment_count": 47
}
```

They can be reconsidered later if the UI has a clear source-metadata surface.

## Backend Gaps To Implement

### 1. Agent Name

The frontend needs a display name for the agent that generated or posted the story. This is separate from `source_name`.

Add this field to `PostListItem` and `PostDetail`:

```json
{
  "agent_name": "HN Tech Agent"
}
```

Recommended v1 implementation:

- Add `agent_name` to `GeneratedPost`.
- Set it during post generation.
- Use a simple default such as `NextNews Agent` or `HN Tech Agent`.
- Do not add an `agents` table until there are multiple configurable agents.

### 2. Single-User App Likes

Likes should be owned by NextNews, not by Hacker News or any upstream source.

For now, the app has one implicit user: the owner of the app. That means `liked_by_me` is still valid, but the backend does not need full registration or authentication yet.

Add these fields to `PostListItem` and `PostDetail`:

```json
{
  "app_like_count": 1,
  "liked_by_me": true
}
```

For the single-user version:

- `app_like_count` is `1` when the implicit user liked the post.
- `app_like_count` is `0` when the implicit user has not liked the post.
- `liked_by_me` is the same single-user liked state.

### 3. Like And Unlike Endpoints

Add endpoints for the frontend to persist likes:

```http
POST /posts/{post_id}/like
DELETE /posts/{post_id}/like
```

Suggested behavior:

- `POST` marks the post as liked.
- `DELETE` removes the like.
- Both endpoints are idempotent.
- Both endpoints return the updated post interaction state.
- If the post does not exist or is not `ready`, return `404`.

Suggested response:

```json
{
  "post_id": 4,
  "app_like_count": 1,
  "liked_by_me": true
}
```

## Suggested Database Shape

Add a small interaction table for app-owned post state:

```text
post_interactions
- id
- post_id
- liked
- liked_at
- created_at
- updated_at
```

Constraints:

- `post_id` should reference `generated_posts.id`.
- `post_id` should be unique for now because there is only one implicit user.

Future user-account version:

```text
post_interactions
- id
- post_id
- user_id
- liked
- liked_at
- created_at
- updated_at
```

The future unique constraint should become `(post_id, user_id)`.

## Target Post Response Shape

After the next backend gap work, `GET /posts` and `GET /posts/{post_id}` should return:

```json
{
  "id": 4,
  "title": "HN spotlight: Starring the Computer",
  "description": "A short summary.",
  "content": "Longer generated post body.",
  "image_url": "/images/post-4.png",
  "source_name": "hacker_news",
  "source_item_id": "48796093",
  "source_url": "https://example.com/article",
  "created_at": "2026-07-06T07:52:25.654423",
  "agent_name": "HN Tech Agent",
  "app_like_count": 1,
  "liked_by_me": true
}
```

Keep `image_url` as a relative URL for now. The iOS app can resolve it against its configured API base URL.

## Recommended Backend Work Order

1. Add `agent_name` to the generated post model and response schema.
2. Populate `agent_name` in the generation pipeline with one configured/default agent name.
3. Add `post_interactions` for the implicit single user.
4. Add `app_like_count` and `liked_by_me` to post list/detail responses.
5. Add `POST /posts/{post_id}/like` and `DELETE /posts/{post_id}/like`.
6. Add focused API tests for list/detail like fields and like/unlike endpoint idempotency.
7. Update the iOS frontend to remove category pills and persist likes through the new endpoints.

## Explicit Non-Goals For This Pass

Do not add these yet:

- Category fields or category filters
- Server-side category search
- Hacker News score/comment fields in the frontend response
- User registration or authentication
- Multi-user like counts
- Agent tables or agent management endpoints
- UI presentation fields such as SF Symbols, colors, badges, or card layout hints
