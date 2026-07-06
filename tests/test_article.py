import asyncio

from app.article import extract_article_text, fetch_article_text


class FakeArticleResponse:
    def __init__(self, text: str, content_type: str = "text/html") -> None:
        self.text = text
        self.headers = {"content-type": content_type}

    def raise_for_status(self) -> None:
        return None


class FakeArticleClient:
    def __init__(self, response: FakeArticleResponse) -> None:
        self.response = response
        self.requests: list[dict] = []

    async def get(self, url: str, *, follow_redirects: bool) -> FakeArticleResponse:
        self.requests.append({"url": url, "follow_redirects": follow_redirects})
        return self.response


def test_extract_article_text_removes_non_visible_html() -> None:
    text = extract_article_text(
        """
        <html>
          <head><style>.hidden { display: none; }</style></head>
          <body>
            <nav>Navigation</nav>
            <article>
              <h1>Launch update</h1>
              <p>The team shipped a faster model.</p>
              <script>window.secret = true</script>
              <p>It reduces waiting time for developers.</p>
            </article>
          </body>
        </html>
        """
    )

    assert "display: none" not in text
    assert "window.secret" not in text
    assert "Launch update" in text
    assert "The team shipped a faster model." in text
    assert "It reduces waiting time for developers." in text


def test_fetch_article_text_reads_html() -> None:
    client = FakeArticleClient(
        FakeArticleResponse("<article><p>Readable article body.</p></article>")
    )

    text = asyncio.run(fetch_article_text(client, "https://example.com/article"))

    assert text == "Readable article body."
    assert client.requests == [
        {"url": "https://example.com/article", "follow_redirects": True}
    ]
