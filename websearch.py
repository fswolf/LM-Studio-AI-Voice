"""
websearch.py

Lets you say things like "agent please web search current bitcoin
price" and get real, live results folded into the answer for that
turn - without depending on the model's own tool/function-calling
support.

Why not native tool calling: LM Studio can expose tool calls to
models that support it, but that only works if the model reliably
emits well-formed tool_call JSON on every request. Heavily fine-tuned
"uncensored" merges are often inconsistent at that. The same
cheap-keyword-gate pattern already used in reminders.py is more
robust here: a plain substring check decides whether to search at
all - no extra model call just to decide - then real results get
folded straight into the prompt for this turn.

How it works:
1. A cheap keyword check ("web search" in the message) decides
   whether to search at all.
2. If it passes, the text after the trigger phrase becomes the
   search query and gets run through DuckDuckGo via the `ddgs`
   package (pip install ddgs) - no API key needed.
3. Results are formatted into a compact block and added as an extra
   system-role message for *this turn only*. The user's original
   text is still what gets saved to history via history.add_message()
   in llm.py, so raw search results never pollute the permanent
   conversation log or get repeatedly re-sent on every future turn.

Known limitations, worth knowing:
- Query extraction is a plain text-split, not model-based - it takes
  everything after the trigger phrase as the query. Works well for a
  direct phrasing like "web search X", less well for a buried
  request like "so anyway, could you maybe web search X for me".
- Speech-to-text can mangle "web search" into something else
  entirely - if voice triggering feels unreliable, check what Whisper
  actually transcribed before assuming this code is at fault.
- ddgs scrapes DuckDuckGo's result pages rather than using an
  official API, so it can occasionally get rate-limited or return
  nothing. This fails soft (falls back to a normal answer with no
  results) rather than raising or crashing the conversation.
- Needs an internet connection - the one deliberate exception to the
  rest of this project's local/offline design.
"""

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
