import re

from config import WEB_SEARCH_ENABLED, WEB_SEARCH_MAX_RESULTS

_TRIGGER_PATTERN = re.compile(r"\bweb\s*search\b", re.IGNORECASE)

_LEADING_FILLER = re.compile(r"^(for|about|on|up)\b\s*", re.IGNORECASE)
_TRAILING_FILLER = re.compile(r"\b(please|nya|thanks|thank you)\W*$", re.IGNORECASE)


def looks_like_search(text: str) -> bool:
    return bool(WEB_SEARCH_ENABLED and _TRIGGER_PATTERN.search(text))


def extract_query(text: str) -> str:
    """Everything after the trigger phrase, lightly cleaned up.

    e.g. "agent please web search current bitcoin price nya" ->
    "current bitcoin price"
    """
    after = _TRIGGER_PATTERN.split(text, maxsplit=1)[-1].strip()
    after = _LEADING_FILLER.sub("", after)
    after = _TRAILING_FILLER.sub("", after).strip()
    after = after.strip(" ?.!,\"'")
    return after or text.strip()


def search(query: str, max_results: int = None) -> list:
    """Runs a live DuckDuckGo search. Returns [] on any failure
    (missing package, no internet, rate limit, timeout) instead of
    raising, so a flaky search never breaks the conversation - Luna
    just answers without it, or says she couldn't find anything.
    """
    max_results = max_results or WEB_SEARCH_MAX_RESULTS

    try:
        from ddgs import DDGS
    except ImportError:
        print("[websearch] `ddgs` isn't installed - run: pip install ddgs")
        return []

    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        print(f"[websearch] search failed: {e}")
        return []


def format_results(query: str, results: list) -> str:
    """Builds the system-role context block injected for this turn."""
    if not results:
        return (
            f'(A web search for "{query}" was attempted but returned no '
            "results. Tell the user you couldn't find anything for that - "
            "don't make up an answer.)"
        )

    lines = [
        f'(Live web search results for "{query}". Use these to answer '
        "naturally in your own voice and mention you looked it up - "
        "don't read them out verbatim or list every result, just work "
        "the relevant bits into a normal spoken reply.)"
    ]
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        body = (r.get("body") or "").strip()
        url = (r.get("href") or "").strip()
        lines.append(f"{i}. {title} - {body} ({url})")
    return "\n".join(lines)
