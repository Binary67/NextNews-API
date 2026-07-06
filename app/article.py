import logging
from html.parser import HTMLParser

import httpx
import trafilatura


logger = logging.getLogger(__name__)


class HtmlSnippetTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
            return

        if tag in {"article", "section", "div", "p", "br", "li", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return

        if tag in {"article", "section", "div", "p", "li", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return

        text = " ".join(data.split())
        if text:
            self._parts.append(text)

    def text(self) -> str:
        return normalize_text(" ".join(self._parts))


def normalize_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_html_snippet_text(document: str) -> str:
    parser = HtmlSnippetTextParser()
    parser.feed(document)
    parser.close()
    return parser.text()


def extract_article_text(document: str) -> str | None:
    text = trafilatura.extract(
        document,
        output_format="txt",
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    if text is None:
        return None

    return normalize_text(text) or None


async def fetch_article_text(client: httpx.AsyncClient, url: str) -> str | None:
    response = await client.get(url, follow_redirects=True)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if content_type.startswith("text/plain"):
        return normalize_text(response.text) or None

    if content_type and not content_type.startswith("text/html"):
        return None

    logger.info("Extracting article text from %s", url)
    article_text = extract_article_text(response.text)
    if article_text:
        logger.info(
            "Extracted article text from %s (%s chars)",
            url,
            len(article_text),
        )
    else:
        logger.info("Article text extraction returned no content for %s", url)

    return article_text
