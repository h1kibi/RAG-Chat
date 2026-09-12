"""Lexical complement for dense retrieval candidates.

Dense-only ranking misses exact technical identifiers (``CVE-2021-43798``,
``JNDI``, ``rsync``) when the surrounding prose is generic: a passage *about*
exploitation sits closer to the query vector than the specific service writeup
does.  This module scores lexical evidence and blends it with the cosine score::

    final = (1 - weight) * dense + weight * lexical

Tokenization is intentionally dependency-free and query-local:

- ASCII/latin tokens (lower-cased), e.g. ``grafana``, ``cve-2021-43798``;
- character bigrams over the CJK substring, e.g. ``任意文件读取`` ->
  ``任意`` ``意文`` ``文件`` ``件读`` ``读取``.

The lexical value is produced by :class:`LexicalReranker` over the whole
candidate pool, using local inverse document frequency and a separate signal for
a match in the source path.  A plain hit ratio cannot work here: with one
identifier in the query its ceiling is well below the generic tokens' combined
weight, so raising ``weight`` reorders nothing — the exact-match document loses
by a hair at every setting.
"""
from __future__ import annotations

import math
import re
import zlib
from typing import Any, Sequence

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9._+:-]*", re.IGNORECASE)

IDENTIFIER_MIN_LENGTH = 4
"""Shortest token indexed for exact-identifier recall (``873`` is too generic)."""


def identifier_tokens(text: str) -> set[str]:
    """Return the digit-bearing tokens worth exact lookup.

    Querying ``CVE-2021-3490`` is the canonical "user supplies only an
    identifier" case, and dense retrieval can bury the one document that states
    it: the term is rare, so no amount of semantic neighbourhood helps. These
    tokens are indexed offline so a match bypasses dense ranking entirely.

    Restricting to digit-bearing tokens keeps the index small (~30 MB for 1M
    chunks) because the tail is dominated by one-off identifiers — which is
    exactly what makes them worth indexing. Purely alphabetic tokens are left to
    the dense stage and the source-path signal, which already handle them.
    """
    if not text:
        return set()
    found = set()
    for token in _WORD_RE.findall(text.lower()):
        if len(token) >= IDENTIFIER_MIN_LENGTH and any(ch.isdigit() for ch in token):
            found.add(token)
    return found


_PATH_WEIGHT = 0.55
_BODY_WEIGHT = 0.45
_IDF_EXPONENT = 2.0
"""Sharpens inverse document frequency so rare tokens stay decisive.

A plain sum lets three generic tokens (``漏洞`` ``利用`` ``提权``) together
outweigh the one discriminating token (``rsync``) a document actually matches,
which is precisely the ranking this module exists to prevent. Squaring the
frequency ratio is the standard remedy (the same idea as BM25's idf exponent):
a token in most documents contributes ~0, a token in a handful dominates.
"""


class QueryTerms:
    """Precomputed token sets for one query."""

    __slots__ = ("ascii_tokens", "cjk_bigrams", "total", "ascii_patterns")

    def __init__(self, query: str):
        ascii_tokens = [token.lower() for token in _WORD_RE.findall(query) if len(token) >= 2]
        # Bigrams are built per contiguous Hanzi run. Joining the runs first
        # would fabricate cross-boundary pairs ("漏洞 利用" -> 洞利) that match
        # no document, yet still consume IDF weight and dilute real hits.
        bigrams: list[str] = []
        for run in _CJK_RUN_RE.findall(query):
            if len(run) == 1:
                continue
            for index in range(len(run) - 1):
                bigram = run[index : index + 2]
                if bigram not in bigrams:
                    bigrams.append(bigram)
        self.ascii_tokens = ascii_tokens
        self.cjk_bigrams = bigrams
        self.total = len(ascii_tokens) + len(bigrams)
        # An ASCII token must not match inside a longer word: without this,
        # "rsync" matches "UserSynchronization" and the IDF signal rewards
        # unrelated documents. Hanzi is excluded from the boundary class so
        # "的rsync的" still matches, unlike a plain ``\b``.
        self.ascii_patterns = [
            re.compile(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])")
            for token in ascii_tokens
        ]

    def matches_ascii(self, lowered_text: str) -> int:
        """Count ASCII tokens present as standalone words."""
        return sum(1 for pattern in self.ascii_patterns if pattern.search(lowered_text))

    @property
    def is_empty(self) -> bool:
        return self.total == 0

    def tokens(self) -> list[str]:
        """All query tokens; identifiers first, then Hanzi bigrams."""
        return [*self.ascii_tokens, *self.cjk_bigrams]


def tokenize_query(query: str) -> QueryTerms:
    """Return query terms; raises for blank input only."""
    if not query or not query.strip():
        raise ValueError("query must not be blank")
    return QueryTerms(query)


class LexicalReranker:
    """Score a whole candidate pool with IDF-weighted, field-aware hits.

    Document frequency is measured *within the pool* dense retrieval already
    selected, so no global term index and no extra corpus pass is needed:

    - a token present in few candidates carries the most weight, which is what
      makes ``rsync``/``TCC`` outrank ``漏洞``/``利用``;
    - a token in the source path outweighs the same token in the body, because
      corpus file names are topic labels (``873-pentesting-rsync.md``).
    """

    __slots__ = ("terms", "scores")

    def __init__(self, terms: QueryTerms, candidates: Sequence[tuple[str, str]]):
        self.terms = terms
        self.scores: list[float] = [0.0] * len(candidates)
        if terms.is_empty or not candidates:
            return
        bodies = [(text or "").lower() for text, _ in candidates]
        paths = [(source or "").replace("\\", "/").lower() for _, source in candidates]
        tokens = terms.tokens()
        weights: list[float] = []
        for index, token in enumerate(tokens):
            if index < len(terms.ascii_tokens):
                pattern = terms.ascii_patterns[index]
                document_frequency = sum(1 for body in bodies if pattern.search(body))
                path_frequency = sum(1 for path in paths if pattern.search(path))
            else:
                document_frequency = sum(1 for body in bodies if token in body)
                path_frequency = sum(1 for path in paths if token in path)
            # A token present nowhere keeps the largest weight (its absence is
            # what makes it discriminating); a token present everywhere gets
            # ~0 and therefore cannot decide the order.
            weights.append(
                math.log(
                    (len(candidates) + 1.0)
                    / (document_frequency + path_frequency + 0.5)
                )
                ** _IDF_EXPONENT
            )
        total = sum(weights)
        if total <= 1e-9:
            return
        inverse = 1.0 / total
        ascii_count = len(terms.ascii_tokens)
        for index in range(len(candidates)):
            body_hits = 0.0
            path_hits = 0.0
            for position, (token, weight) in enumerate(zip(tokens, weights)):
                if position < ascii_count:
                    pattern = terms.ascii_patterns[position]
                    if pattern.search(bodies[index]):
                        body_hits += weight
                    if pattern.search(paths[index]):
                        path_hits += weight
                else:
                    if token in bodies[index]:
                        body_hits += weight
                    if token in paths[index]:
                        path_hits += weight
            self.scores[index] = min(
                1.0,
                _BODY_WEIGHT * body_hits * inverse + _PATH_WEIGHT * path_hits * inverse,
            )


def fusion_score(dense: float, lexical: float, weight: float) -> float:
    """Blend dense cosine and lexical score into a stable final score."""
    dense_value = max(0.0, float(dense))
    if weight <= 0.0:
        return dense_value
    return (1.0 - weight) * dense_value + weight * max(0.0, min(1.0, float(lexical)))


_LINK_LINE_RE = re.compile(r"^(?:[-*]\s+)?(?:https?://|\[[^\]]+\]\(https?://)", re.IGNORECASE)


def low_information_reasons(text: str) -> list[str]:
    """Return flags when a chunk is likely low-information padding.

    Conservative, explainable heuristics only; a normal paragraph never
    triggers all three:

    - ``repetitive``: one long run built from very few unique characters
      (``testtesttest...``-style pollution or filler);
    - ``sparse``: a long chunk that is almost entirely whitespace, i.e. a
      handful of characters padded out to full chunk size (binary and
      attachment dumps chunk this way);
    - ``numeric``: a long chunk with no letter or Hanzi at all — coordinate
      tables and debug parameter dumps, never an explanation;
    - ``link-heavy``: at least 60% of non-empty lines are bare URLs or
      markdown links (reference/footer sections);
    - ``no-text``: no letter, digit, or CJK character at all.
    """
    if not text or not text.strip():
        return ["empty"]
    reasons: list[str] = []
    compact = "".join(text.split())
    if len(compact) >= 120 and len(set(compact)) <= 12:
        reasons.append("repetitive")
    if len(text) >= 200 and len(compact) < len(text) * 0.15:
        reasons.append("sparse")
    if len(text) >= 100 and not re.search(r"[A-Za-z\u4e00-\u9fff]", text):
        reasons.append("numeric")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        link_lines = sum(1 for line in lines if _LINK_LINE_RE.match(line))
        if link_lines / len(lines) >= 0.6:
            reasons.append("link-heavy")
    if not re.search(r"[0-9A-Za-z\u4e00-\u9fff]", text):
        reasons.append("no-text")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    blob_lines = 0
    for line in lines:
        if len(line) >= 64 and re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", line):
            blob_lines += 1
    if len(lines) >= 3 and blob_lines / len(lines) >= 0.5:
        reasons.append("blob-heavy")
    return reasons


_IMAGE_MD_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")

_WHITESPACE_RE = re.compile(r"\s+")
_MINHASH_PRIME = (1 << 61) - 1
_MINHASH_A = (1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
_MINHASH_B = (41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89)
_GRAM_WIDTH = 8


def text_signature(text: str) -> list[int]:
    """Return a 12-value minhash signature over whitespace-free character
    n-grams; used to detect near-duplicate chunks across mirrored sources.

    Each gram is hashed once into a 64-bit value, then permuted per row. The
    per-row projection must be able to wrap around ``_MINHASH_PRIME``: a 32-bit
    gram hash cannot, so ``factor_a * hash + factor_b`` stays monotone in
    ``hash`` and every row would select the same globally-smallest gram. That
    degenerate signature is not a Jaccard estimate — it only compares the
    single smallest-hash gram, and a mirror copy that merely prepends a header
    line ("本文首发于…") scored as a completely unrelated document.
    """
    normalized = _WHITESPACE_RE.sub("", text or "").lower()
    length = len(normalized)
    if length < _GRAM_WIDTH:
        return [0] * len(_MINHASH_A)
    gram_hashes = set()
    for start in range(length - _GRAM_WIDTH + 1):
        encoded = normalized[start : start + _GRAM_WIDTH].encode("utf-8")
        # crc32 and adler32 are independent enough to fill 64 bits; both are
        # C-implemented and stable across processes.
        combined = (zlib.crc32(encoded) << 32) | zlib.adler32(encoded)
        gram_hashes.add(combined % _MINHASH_PRIME)
    return [
        min((factor_a * value + factor_b) % _MINHASH_PRIME for value in gram_hashes)
        for factor_a, factor_b in zip(_MINHASH_A, _MINHASH_B)
    ]


def signature_similarity(left: list[int], right: list[int]) -> float:
    """Return the minhash Jaccard estimate in ``[0, 1]``."""
    if not left or len(left) != len(right):
        return 0.0
    matches = sum(1 for a, b in zip(left, right) if a == b)
    return matches / len(left)


def is_near_duplicate(left: list[int], right: list[int], threshold: float = 0.85) -> bool:
    return signature_similarity(left, right) >= threshold


def strip_markdown_images(text: str) -> str:
    """Remove markdown image syntax, keeping any alt text."""
    if not text or "![" not in text:
        return text
    return _IMAGE_MD_RE.sub(lambda match: match.group(1).strip(), text)


_GLIBC_RE = re.compile(r"\bglibc[\s\-_]*(2\.\d{1,2})\b", re.IGNORECASE)
_LIBC_ALT_RE = re.compile(r"\blibc[\s\-_]*(2\.\d{1,2})\b", re.IGNORECASE)
_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
_ARCH_PATTERNS = (
    ("amd64", re.compile(r"\bx86[_-]?64\b|\bamd64\b", re.IGNORECASE)),
    ("i386", re.compile(r"\bi[3-6]86\b|\bx86\b(?![\s_-]?64)", re.IGNORECASE)),
    ("arm64", re.compile(r"\barm64\b|\baarch64\b", re.IGNORECASE)),
    ("arm", re.compile(r"\barm(?:v[3-8]\w*|el|hf)\b", re.IGNORECASE)),
    ("mips", re.compile(r"\bmips\w*\b", re.IGNORECASE)),
    ("windows", re.compile(r"\bwindows\b|\bwin32\b|\bwin64\b", re.IGNORECASE)),
    ("macos", re.compile(r"\bmacos\b|\bosx\b|\bdarwin\b", re.IGNORECASE)),
    ("linux", re.compile(r"\blinux\b", re.IGNORECASE)),
)


def extract_environment_facts(text: str, limit: int = 6) -> dict[str, Any]:
    """Return version/architecture facts stated by the evidence itself.

    The reviewer's point stands: a 2.31 writeup returned for a 2.35 target is a
    silent hazard, and a warning in the tool description relies on the agent
    reading it. Reporting the versions the *document* actually mentions lets a
    caller compare them against the target environment mechanically.

    Only literal observations are returned — no inference about which version a
    document "targets". A document stating no version yields no ``glibc`` key
    rather than a guess.
    """
    if not text:
        return {}
    facts: dict[str, Any] = {}
    versions = {match.group(1) for match in _GLIBC_RE.finditer(text)}
    versions |= {match.group(1) for match in _LIBC_ALT_RE.finditer(text)}
    if versions:
        facts["glibc"] = sorted(versions, key=lambda value: [int(p) for p in value.split(".")])
    cves = {match.group(0).upper() for match in _CVE_RE.finditer(text)}
    if cves:
        facts["cve"] = sorted(cves)[:limit]
    architectures = [
        name for name, pattern in _ARCH_PATTERNS if pattern.search(text)
    ]
    if architectures:
        facts["arch"] = architectures
    return facts


def screenshot_placeholders(text: str) -> int:
    """Count image references that stand in for evidence.

    ``strip_images`` keeps alt text, so a writeup whose key (a cracked key, a
    flag, a byte layout) is visible only inside a screenshot degrades into
    "Pasted image 20251104202924.png" with no way for the caller to tell the
    content is missing rather than absent. Counting the placeholders lets the
    result advertise that it must fall back to the original file.
    """
    if not text:
        return 0
    return len(_IMAGE_MD_RE.findall(text))


_PROVENANCE_KEYS = (
    "source_repository",
    "source_url",
    "source_path",
    "source_commit",
    "retrieved_at",
    "usage",
    "notice",
)


def parse_provenance_frontmatter(text: str) -> dict[str, str]:
    """Return the importer's provenance fields when the block leads ``text``.

    Retrieval strips this block from returned content because it is identical
    across every chunk of a document and wastes tokens. The fields still answer
    real questions — which upstream repository a claim came from, at which
    commit, and when it was retrieved — so callers keep them as metadata rather
    than deleting them. Blocks that open with ``---`` but carry no known
    provenance key are author-owned frontmatter and yield nothing.
    """
    if not text or not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end <= 0 or end > 4096:
        return {}
    header = text[4:end]
    if not any(f"{key}:" in header for key in _PROVENANCE_KEYS):
        return {}
    fields: dict[str, str] = {}
    for line in header.splitlines():
        name, separator, value = line.partition(":")
        if not separator:
            continue
        name = name.strip()
        if name in _PROVENANCE_KEYS and value.strip():
            fields[name] = value.strip()
    return fields


def strip_provenance_frontmatter(text: str) -> str:
    """Remove the importer's YAML provenance block if it leads the chunk.

    Only blocks that open with ``---`` and contain at least one known
    provenance key are removed, so author-owned frontmatter in the corpus is
    never touched. Use :func:`parse_provenance_frontmatter` first when the
    fields are wanted as metadata.
    """
    if not parse_provenance_frontmatter(text):
        return text
    end = text.find("\n---\n", 4)
    return text[end + 5 :].lstrip("\n")


def snippet_window(text: str, query: str, size: int) -> str:
    """Return a bounded window around the first query-token hit.

    Without a hit the window starts at the top of the chunk. Window edges are
    expanded to whole lines (limited look-around) so agents do not see
    mid-sentence fragments; truncation is marked with ``…``.
    """
    if size <= 0 or not text:
        return text
    if len(text) <= size:
        return text
    position = 0
    terms = tokenize_query(query) if query and query.strip() else QueryTerms("")
    if not terms.is_empty:
        lowered = text.lower()
        best: int | None = None
        for token in terms.ascii_tokens:
            index = lowered.find(token)
            if index >= 0 and (best is None or index < best):
                best = index
        if best is None:
            for bigram in terms.cjk_bigrams:
                index = text.find(bigram)
                if index >= 0 and (best is None or index < best):
                    best = index
        if best is not None:
            position = best
    half = max(1, size // 2)
    start = max(0, position - half)
    end = min(len(text), start + size)
    # expand to line boundaries without overshooting the budget too far
    look = min(start, 80)
    while start > 0 and text[start - 1] != "\n" and look > 0:
        start -= 1
        look -= 1
    look = min(len(text) - end, 80)
    while end < len(text) and text[end] != "\n" and look > 0:
        end += 1
        look -= 1
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    window = prefix + text[start:end].rstrip() + suffix
    if window.strip("…").strip():
        return window
    # A blank window must never stand in for a non-blank document: the caller
    # cannot tell "this chunk is empty" from "the window collapsed", and would
    # read it as absent evidence. Fall back to a bounded, marked prefix.
    return text[: max(size, 32)].rstrip() + "…"


# --- Evidence extraction -----------------------------------------------------
#
# A caller who wants the payload out of a writeup currently receives a
# query-centred text window and has to re-find and re-assemble the code by hand.
# These helpers hand back the discrete segments plus their provenance, so the
# extraction happens once, in the layer that already has the document.

_HTML_IMG_RE = re.compile(r"<img\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)
_FENCE_RE = re.compile(
    r"^[ \t]*(`{3,}|~{3,})[ \t]*([^\n`]*?)\r?\n(.*?)(?:^[ \t]*\1[ \t]*\r?$)",
    re.M | re.S,
)

_PAYLOAD_LANGUAGES = frozenset(
    {"python", "py", "bash", "sh", "shell", "zsh", "powershell", "ps1", "cmd",
     "c", "cpp", "c++", "csharp", "java", "javascript", "js", "typescript", "ts",
     "php", "ruby", "rb", "perl", "go", "rust", "sql", "lua", "asm", "nasm",
     "http", "yaml", "yml", "json", "xml", "ini", "conf", "dockerfile", "makefile"}
)
"""Fence languages treated as executable/material rather than prose or output.

An unlabelled fence is still returned (``kind="code"``): writeups routinely omit
the language on the one block that matters, and dropping those would lose the
exact segment the caller asked for. ``kind="payload"`` is the stricter view and
requires a language that denotes runnable material or a payload-shaped body.
"""

_PAYLOAD_HINT_RE = re.compile(
    r"(?im)^\s*(?:\$|#|>|PS[ >])|curl\s|wget\s|nc\s+-|ncat\s|python[23]?\s+-c|"
    r"perl\s+-e|ruby\s+-e|import\s+\w+|from\s+\w+\s+import|"
    r"(?:get|post|put|delete)\s+/[^\s]*\s+HTTP/|"
    r"[\w./-]+\.(?:py|sh|rb|pl|go|c|cpp|java|php)\b"
)


_IMAGE_LINK_RE = re.compile(
    r"!\[([^\]]*)\]\(\s*<?([^\s)>\x22\x27]*)>?(?:\s+[\x22\x27][^\x22\x27]*[\x22\x27])?\s*\)"
)
_UNSAFE_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*:", re.IGNORECASE)


def _safe_image_target(src: str) -> str | None:
    """Return the address if it is fetchable, else ``None``.

    Only http(s) and relative paths qualify. A corpus document is untrusted
    input, so a `javascript:` or `data:` "image address" must not be handed to a
    caller that might act on it -- and returning it in a field named `image_refs`
    invites exactly that. The previous parse also produced garbage for malformed
    markdown, which is worse than nothing because it looks like a real path.
    """
    src = (src or "").strip()
    if not src or any(ch in src for ch in " \t\r\n\"'<>"):
        return None
    if _UNSAFE_SCHEME_RE.match(src):
        return src if src[:7].lower() == "http://" or src[:8].lower() == "https://" else None
    return src


def image_references(text: str) -> list[dict[str, str]]:
    """Return the images a chunk embeds, so a caller can fetch them itself.

    ``strip_images`` replaces the syntax with its alt text, which is right for a
    text-only model but destroys the address. A multimodal caller can act on the
    screenshot; it just needs the URL, which is why ``shots=N`` alone is not
    enough.
    """
    if not text:
        return []
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in _IMAGE_LINK_RE.finditer(text):
        src = _safe_image_target(match.group(2))
        if not src or src in seen:
            continue
        seen.add(src)
        found.append({"src": src, "alt": match.group(1).strip()})
    for match in _HTML_IMG_RE.finditer(text):
        src = _safe_image_target(match.group(1))
        if src and src not in seen:
            seen.add(src)
            found.append({"src": src, "alt": ""})
    return found


def extract_segments(text: str, kind: str = "code", max_segments: int = 8) -> list[dict[str, Any]]:
    """Pull fenced segments out of a chunk, labelled with their language.

    ``kind="code"`` returns every fenced block. ``kind="payload"`` keeps only
    blocks that look runnable -- a language naming a programming or shell
    language, or a body shaped like a command/exploit. Both are ordered by
    position in the document.
    """
    if not text or kind not in {"code", "payload"}:
        return []
    out: list[dict[str, Any]] = []
    for match in _FENCE_RE.finditer(text):
        body = match.group(3)
        if body is None:
            continue
        body = body.rstrip("\r\n")
        if not body.strip():
            continue
        language = (match.group(2) or "").strip().split()[0].lower() if match.group(2) else ""
        if kind == "payload":
            if not (language in _PAYLOAD_LANGUAGES or _PAYLOAD_HINT_RE.search(body)):
                continue
        out.append({"language": language or None, "text": body})
        if len(out) >= max_segments:
            break
    return out


# --- Past-event flags --------------------------------------------------------
#
# Writeups quote the flag they captured. That is useful for reproducing the
# original challenge and actively misleading for a variant: a solver that has
# the flag shown to it will try to make it fit. The service cannot know which
# case it is in, so it marks what it found and leaves the judgement to the
# caller. Marking, not removing -- redaction would break the reproduction case.
#
# Measured over 60k corpus rows: 2983 candidates, dominated by `flag{...}` (934)
# and event names (`dasctf`, `actf`, `ductf`, `lactf`, ...). The only common
# false positives are language keywords, listed below.

_FLAG_CANDIDATE_RE = re.compile(r"(?<!\\)\b([A-Za-z][A-Za-z0-9_]{1,23})\{([^{}\n]{8,120})\}")

_CODE_PREFIXES = frozenset({
    "if", "else", "elif", "for", "while", "switch", "case", "catch", "finally",
    "try", "do", "return", "function", "func", "def", "class", "struct", "union",
    "enum", "template", "namespace", "foreach", "with", "lambda", "match",
    "default", "import", "from", "new", "let", "var", "const", "identifier",
    "printf", "sprintf", "format", "echo", "select", "insert", "update", "delete",
    "where", "table", "index", "create", "alter", "values", "print", "range",
    "input", "output", "begin", "end", "text", "label", "ref", "cite", "section",
    "sleep", "success", "error", "debug", "info", "warn", "trace", "stdout",
})
"""Prefixes that are syntax rather than an event name.

Measured over 60k corpus rows: `else` and `try` were the only frequent ones (117
and 23); `\begin{document}`-style LaTeX and `input{...}` were the other visible
false positives, which is why backslash-preceded candidates and these names are
excluded.
"""

_FLAG_BODY_RE = re.compile(r"^[A-Za-z0-9_!@#$%^&*\-+=:.?~]{8,}$")
_PLAIN_WORD_RE = re.compile(r"^[a-z]+$")
"""A body with no uppercase, digit or punctuation, and short, is prose or code.

`processes` and `document` were the observed false positives; real flags in the
survey either mix case (`YouKnowHowToFuzz!`), carry digits (`12qwaszxcde3`) or
use punctuation (`_tihne__ifnlfaign_igtoyt`).
"""


def past_event_flags(text: str, limit: int = 6) -> list[str]:
    """Flag-shaped strings a document quotes, for the caller to sanity-check.

    Tuned for recall: a flag this misses is a solver that silently copies a
    past-event answer, while a code snippet marked by mistake only adds a line
    of caution. Language keywords and bodies that are plainly source code are
    filtered because they were the measured false positives.
    """
    if not text or "{" not in text:
        return []
    out: list[str] = []
    for match in _FLAG_CANDIDATE_RE.finditer(text):
        prefix, body = match.group(1), match.group(2)
        if prefix.lower() in _CODE_PREFIXES:
            continue
        if not _FLAG_BODY_RE.match(body):
            continue
        if _PLAIN_WORD_RE.match(body):
            continue
        candidate = f"{prefix}{{{body}}}"
        if candidate not in out:
            out.append(candidate)
        if len(out) >= limit:
            break
    return out


# --- Provenance tier ---------------------------------------------------------
#
# The corpus mixes maintained reference material, competition writeups and
# community blog mirrors, and they ranked indistinguishably. Reported from agent
# use: "镜像博客（先知/补天）、HackTricks、赛事 writeup 平权混排".
#
# This is a factual statement about where a source came from, not a quality
# score. A blog mirror can be the better writeup; the caller just deserves to
# know which kind of thing it is holding. Derived from the first path segment,
# which build_cosine records as the source's category.

_ORIGIN_BY_CATEGORY = {
    "08_ctf_des_knowledge": "handbook",
    "09_hacktricks": "handbook",
    "10_payloads_all_the_things": "handbook",
    "11_lolbas": "handbook",
    "12_security_learning": "blog-mirror",
    "13_xianzhi": "blog-mirror",
    "15_butian": "blog-mirror",
    "14_ctf_wp": "writeup",
}
"""Category → provenance kind.

The curated `00_`-`07_` templates are the repository's own seed documents, so
they are `curated` rather than any of the imported kinds.
"""


def source_origin(source: str | None) -> str | None:
    """Classify a source path by where it came from, or ``None`` if unknown."""
    if not source:
        return None
    category = str(source).replace("\\", "/").split("/", 1)[0]
    if category in _ORIGIN_BY_CATEGORY:
        return _ORIGIN_BY_CATEGORY[category]
    if category[:2].isdigit() and int(category[:2]) < 8:
        return "curated"
    return None
