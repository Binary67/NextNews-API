# NextNews API

Local FastAPI pipeline for AI-generated NextNews posts.

The app ingests Hacker News best stories, stores source items in SQLite, extracts readable article text, filters low-quality sources with Azure OpenAI, generates social-feed-style post content, generates a PNG image, and exposes ready posts through a small read API.

## Requirements

- Python 3.11+
- Azure OpenAI deployments for:
  - LLM post generation
  - LLM source quality filtering
  - Image generation

## Setup

Create and activate a virtual environment, then install the package with development dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Create a local environment file:

```sh
cp .env.example .env
```

Update `.env` with the Azure OpenAI endpoints, API keys, and deployment names for your environment.

## Configuration

Settings are loaded from `.env` by `app.config.Settings`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///./data/nextnews.db` | SQLAlchemy database URL. SQLite is the current default. |
| `AZURE_OPENAI_LLM_ENDPOINT` | empty | Azure OpenAI endpoint/base URL used by the OpenAI Responses client. |
| `AZURE_OPENAI_LLM_API_KEY` | empty | API key for LLM requests. |
| `AZURE_OPENAI_LLM_DEPLOYMENT` | empty | Deployment used to generate post content. |
| `AZURE_OPENAI_QUALITY_FILTER_DEPLOYMENT` | empty | Deployment used to accept or reject source items. |
| `AZURE_OPENAI_IMAGE_ENDPOINT` | empty | Azure image resource endpoint. |
| `AZURE_OPENAI_IMAGE_API_KEY` | empty | API key for image generation. |
| `AZURE_OPENAI_IMAGE_DEPLOYMENT` | empty | Image generation deployment name. |
| `AZURE_OPENAI_IMAGE_API_VERSION` | `2025-04-01-preview` | Azure image generation API version. |
| `POST_GENERATION_LIMIT` | `1` | Maximum source items to evaluate per pipeline pass. |
| `POST_GENERATION_INTERVAL_SECONDS` | `60` | Delay between background pipeline passes. |
| `HN_REFRESH_INTERVAL_SECONDS` | `1800` | Normal interval between Hacker News refresh attempts. |
| `HN_STORY_SCAN_LIMIT` | `500` | Number of Hacker News best-story IDs to scan. |
| `SOURCE_BACKLOG_TARGET` | `100` | Target number of queued source items without generated posts. |
| `SOURCE_BACKLOG_LOW_WATERMARK` | `10` | Backlog threshold that can trigger an early Hacker News refresh. |
| `HN_MAX_ITEM_FETCHES_PER_REFRESH` | `100` | Maximum unseen Hacker News items fetched during one refresh. |
| `IMAGE_OUTPUT_DIR` | `./data/images` | Directory where generated PNG images are written. |
| `IMAGE_QUALITY` | `medium` | Azure image quality: `low`, `medium`, `high`, or `auto`. |
| `IMAGE_SIZE` | `1024x1024` | Requested image size. |

If the full Azure OpenAI configuration is not present, the pipeline still runs but skips post generation.

## Running

Start the API with the background ingestion and generation pipeline:

```sh
uvicorn app.main:app --reload
```

Start the API without the background pipeline:

```sh
uvicorn app.read_only:app --reload
```

The read-only entrypoint is useful for serving existing posts without fetching new Hacker News items or generating new content.

FastAPI's interactive API documentation is available at `/docs` when the server is running.

## API

### `GET /health`

Returns basic service health:

```json
{"status": "ok"}
```

### `GET /posts`

Returns ready posts only. Supports pagination with `limit` and `offset`.

Query parameters:

- `limit`: 1 to 100, default `20`
- `offset`: 0 or greater, default `0`

Each post includes generated content, source metadata, an optional `/images/...` URL, agent name, and app-local like state.

### `GET /posts/{post_id}`

Returns one ready post. Missing, failed, filtered, or still-processing posts return `404`.

### `POST /posts/{post_id}/like`

Marks a ready post as liked. The operation is idempotent and returns:

```json
{
  "post_id": 1,
  "app_like_count": 1,
  "liked_by_me": true
}
```

### `DELETE /posts/{post_id}/like`

Removes the local like from a ready post. The operation is idempotent and returns the updated interaction state.

### `GET /images/{image_name}`

Serves generated images from `IMAGE_OUTPUT_DIR`.

## Data And Runtime Files

- SQLite data is stored at `data/nextnews.db` by default.
- Generated images are stored in `data/images` by default.
- App logs are written to timestamped files in `logs`.
- The app creates current database tables on startup from the SQLAlchemy model definitions.
- Database migrations are not used in this active-development phase.

## Pipeline Behavior

Each background pipeline pass:

1. Refreshes Hacker News source items when the refresh interval or backlog threshold requires it.
2. Deduplicates source items by source and source item ID.
3. Fetches readable article text for source items that have a URL.
4. Uses the quality filter deployment to reject thin, promotional, unsupported, or off-topic items.
5. Uses the LLM deployment to generate post content for accepted items.
6. Uses the image deployment to generate a PNG image.
7. Stores successful posts with status `ready`.

Article fetch failures are stored as failed generated posts. Quality-filter rejections are stored with status `filtered`. Temporary generation failures leave the source item available for a later retry.

## Development

Run the test suite:

```sh
pytest
```

The tests cover the public API, Hacker News ingestion and deduplication, pipeline retry behavior, article extraction, Azure request construction, and logging setup.
