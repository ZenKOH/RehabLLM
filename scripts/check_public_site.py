#!/usr/bin/env python3
"""Zero-dependency static-site checks for the RehabLLM research prototype."""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


class PageInspector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[tuple[str, dict[str, str]]] = []
        self.script_refs: list[str] = []
        self.has_main = False
        self.has_h1 = False
        self.language = ""
        self.evidence_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        if "id" in attributes:
            if attributes["id"] in self.ids:
                raise AssertionError(f"Duplicate id: {attributes['id']}")
            self.ids.add(attributes["id"])
        if tag == "html":
            self.language = attributes.get("lang", "")
        if tag == "main":
            self.has_main = True
        if tag == "h1":
            self.has_h1 = True
        if tag in {"a", "link"}:
            value = attributes.get("href", "")
            if value:
                self.links.append((value, attributes))
        if tag == "script" and attributes.get("src"):
            self.script_refs.append(attributes["src"])
        if tag == "article" and "evidence-card" in attributes.get("class", "").split():
            self.evidence_count += 1
            assert attributes.get("data-category"), "Evidence card lacks category"


def inspect_page(path: Path) -> PageInspector:
    parser = PageInspector()
    parser.feed(path.read_text(encoding="utf-8"))
    assert parser.has_main and parser.has_h1, f"Missing page landmark or heading in {path}"
    assert parser.language == "en-GB", f"Language not declared in {path}"
    for href, attrs in parser.links:
        target = urlsplit(href)
        if target.scheme in {"https", "http"}:
            assert target.scheme == "https", f"Insecure external URL: {href}"
            if attrs.get("target") == "_blank":
                rel = set(attrs.get("rel", "").split())
                assert {"noopener", "noreferrer"} <= rel, f"Unsafe external link: {href}"
        elif href.startswith("#"):
            assert href[1:] in parser.ids, f"Broken in-page link: {href}"
        elif href.startswith("/"):
            local = SITE / href.lstrip("/")
            if href == "/":
                local = SITE / "index.html"
            assert local.is_file(), f"Missing static asset or route: {href}"
        else:
            raise AssertionError(f"Unexpected relative link: {href}")
    for src in parser.script_refs:
        assert src.startswith("/"), f"Script must be local: {src}"
        assert (SITE / src.lstrip("/")).is_file(), f"Script missing: {src}"
    return parser


def main() -> None:
    assert (SITE / "index.html").is_file(), "Site entrypoint missing"
    assert (SITE / "_headers").is_file(), "Cloudflare response headers missing"
    index = inspect_page(SITE / "index.html")
    not_found = inspect_page(SITE / "404.html")
    assert index.evidence_count == 4, "The curated evidence notes changed: review content and test"
    assert not_found.evidence_count == 0
    assert "/assets/app.js" in index.script_refs, "Client-side filter missing"
    assert {"main", "evidence", "research", "method", "roadmap", "safety", "evidence-search"} <= index.ids
    js = (SITE / "assets" / "app.js").read_text(encoding="utf-8")
    assert "aria-pressed" in js and "textContent" in js and "hidden" in js
    content = (SITE / "index.html").read_text(encoding="utf-8")
    assert "Not a medical chatbot" in content
    assert "There is no public chat service or patient-data collection" in content
    assert "https://www.nice.org.uk/guidance/ng236/chapter/Recommendations" in content
    assert "https://www.who.int/news-room/fact-sheets/detail/rehabilitation" in content
    assert "https://pmc.ncbi.nlm.nih.gov/tools/openftlist/" in content
    assert not any(x in content.lower() for x in ('<form', 'stripe.com', 'clerk.com', 'api.openai.com'))
    headers = (SITE / "_headers").read_text(encoding="utf-8")
    assert "Content-Security-Policy:" in headers and "X-Content-Type-Options:" in headers
    print("PASS: 2 valid pages; 4 attributed evidence cards; source links, navigation, safe external links and local assets verified")


if __name__ == "__main__":
    main()
