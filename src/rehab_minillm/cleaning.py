from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

ANCHOR_RE = re.compile(r"\{#[-\w:.]+\}")
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
MULTINEWLINE_RE = re.compile(r"\n{3,}")

BOILERPLATE_HEADINGS: dict[str, re.Pattern[str]] = {
    "competing_interests": re.compile(
        r"^(conflicts? of interest|conflict-of-interest|competing interests?|declaration of interests?)\s*:?$",
        re.I,
    ),
    "author_contributions": re.compile(
        r"^(authors?'? contributions?|author contributions?|contributions?)\s*:?$", re.I
    ),
    "acknowledgements": re.compile(r"^(acknowledg(e)?ments?)\s*:?$", re.I),
    "funding": re.compile(r"^(funding|funding information|financial support)\s*:?$", re.I),
    "data_availability": re.compile(
        r"^(availability of data and materials|data availability|data sharing statement)\s*:?$",
        re.I,
    ),
    "ethics": re.compile(
        r"^(ethics approval|ethical approval|ethics statement|consent for publication)\s*:?$",
        re.I,
    ),
    "references": re.compile(r"^(references|bibliography)\s*:?$", re.I),
}

CONTENT_HEADINGS = re.compile(
    r"^(\d+(?:\.\d+)*[.)]?\s*)?"
    r"(abstract|introduction|background|methods?|materials and methods|results?|discussion|"
    r"limitations?|conclusions?|clinical implications|future work)\s*:?$",
    re.I,
)

INLINE_PATTERNS: dict[str, re.Pattern[str]] = {
    "no_competing_interests": re.compile(
        r"\bthe authors? declare(?:s)? that (?:they|he|she) ha(?:ve|s) no "
        r"(?:competing|conflicting) interests?\.?",
        re.I,
    ),
    "authors_approved": re.compile(
        r"\ball authors? read and approved the final manuscript\.?", re.I
    ),
    "authors_contributed": re.compile(
        r"\ball authors? contributed to (?:the )?(?:study|manuscript)[^.]*\.", re.I
    ),
}


@dataclass(frozen=True)
class CleaningResult:
    text: str
    original_chars: int
    cleaned_chars: int
    removed_chars: int
    removed_sections: dict[str, int]
    removed_inline: dict[str, int]
    anchors_removed: int
    html_tags_removed: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("text", None)
        return payload


def _heading_key(line: str) -> str | None:
    candidate = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s*", "", line.strip())
    candidate = candidate.strip(" #*_-")
    if not candidate or len(candidate) > 100:
        return None
    for key, pattern in BOILERPLATE_HEADINGS.items():
        if pattern.match(candidate):
            return key
    return None


def detect_contamination(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, pattern in BOILERPLATE_HEADINGS.items():
        matches = 0
        for line in text.splitlines():
            candidate = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s*", "", line.strip())
            candidate = candidate.strip(" #*_-")
            if pattern.match(candidate):
                matches += 1
        if matches:
            counts[f"heading:{key}"] = matches

    for key, pattern in INLINE_PATTERNS.items():
        found = len(pattern.findall(text))
        if found:
            counts[f"inline:{key}"] = found

    anchors = len(ANCHOR_RE.findall(text))
    if anchors:
        counts["markup:section_anchor"] = anchors

    html_tags = len(HTML_TAG_RE.findall(text))
    if html_tags:
        counts["markup:html_tag"] = html_tags
    return counts


def clean_document(text: str) -> CleaningResult:
    original = text.replace("\r\n", "\n").replace("\r", "\n")
    removed_sections: dict[str, int] = {}
    kept: list[str] = []
    skipping: str | None = None

    for line in original.splitlines():
        heading = _heading_key(line)
        if heading is not None:
            skipping = heading
            removed_sections[heading] = removed_sections.get(heading, 0) + 1
            continue

        if skipping is not None:
            stripped = line.strip()
            if CONTENT_HEADINGS.match(stripped):
                skipping = None
                kept.append(line)
            else:
                removed_sections[skipping] = removed_sections.get(skipping, 0) + 1
            continue

        kept.append(line)

    cleaned = "\n".join(kept)

    anchors_removed = len(ANCHOR_RE.findall(cleaned))
    cleaned = ANCHOR_RE.sub("", cleaned)

    html_tags_removed = len(HTML_TAG_RE.findall(cleaned))
    cleaned = HTML_TAG_RE.sub(" ", cleaned)

    removed_inline: dict[str, int] = {}
    for key, pattern in INLINE_PATTERNS.items():
        cleaned, count = pattern.subn("", cleaned)
        if count:
            removed_inline[key] = count

    lines = [MULTISPACE_RE.sub(" ", line).strip() for line in cleaned.splitlines()]
    cleaned = MULTINEWLINE_RE.sub("\n\n", "\n".join(lines)).strip()

    return CleaningResult(
        text=cleaned,
        original_chars=len(original),
        cleaned_chars=len(cleaned),
        removed_chars=max(0, len(original) - len(cleaned)),
        removed_sections=removed_sections,
        removed_inline=removed_inline,
        anchors_removed=anchors_removed,
        html_tags_removed=html_tags_removed,
    )
