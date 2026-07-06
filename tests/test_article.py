import asyncio

from app.article import extract_html_snippet_text, fetch_article_text


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


def test_extract_html_snippet_text_removes_non_visible_html() -> None:
    text = extract_html_snippet_text(
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


def test_fetch_article_text_reads_html_with_trafilatura(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    def fake_extract(document: str, **kwargs) -> str:
        calls.append((document, kwargs))
        return "Readable\n\narticle body."

    monkeypatch.setattr("app.article.trafilatura.extract", fake_extract)
    client = FakeArticleClient(
        FakeArticleResponse("<article><p>Ignored parser body.</p></article>")
    )

    text = asyncio.run(fetch_article_text(client, "https://example.com/article"))

    assert text == "Readable\narticle body."
    assert client.requests == [
        {"url": "https://example.com/article", "follow_redirects": True}
    ]
    assert calls == [
        (
            "<article><p>Ignored parser body.</p></article>",
            {
                "output_format": "txt",
                "include_comments": False,
                "include_tables": True,
                "favor_precision": True,
            },
        )
    ]


def test_fetch_article_text_returns_none_when_trafilatura_cannot_extract(
    monkeypatch,
) -> None:
    def fake_extract(document: str, **kwargs) -> None:
        return None

    monkeypatch.setattr("app.article.trafilatura.extract", fake_extract)
    client = FakeArticleClient(
        FakeArticleResponse(
            "<html><body><nav>Navigation</nav><p>Broad parser would read this.</p></body></html>"
        )
    )

    text = asyncio.run(fetch_article_text(client, "https://example.com/article"))

    assert text is None


def test_fetch_article_text_reads_plain_text_without_trafilatura(monkeypatch) -> None:
    def unexpected_extract(document: str, **kwargs) -> None:
        raise AssertionError("trafilatura should not run for text/plain responses")

    monkeypatch.setattr("app.article.trafilatura.extract", unexpected_extract)
    client = FakeArticleClient(
        FakeArticleResponse(" Plain text\n\narticle body. ", content_type="text/plain")
    )

    text = asyncio.run(fetch_article_text(client, "https://example.com/article.txt"))

    assert text == "Plain text\narticle body."
