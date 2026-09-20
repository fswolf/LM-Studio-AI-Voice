"""Reading a page, rather than a search snippet about it.

web_search returns a title, roughly two hundred characters of blurb and
a URL. That is enough to answer "what's the weather" and nowhere near
enough for "what does this article actually say" - so she would
summarize the blurb and sound confident about it, which is the worst of
both.

This fetches the page and strips it to readable text. Stdlib only:
HTMLParser rather than BeautifulSoup, because one more dependency for
"delete the script tags" isn't worth it, and this way it works in the
same venv that's already there.

Everything here is hostile-input handling. A web page is text written
by someone else that the model is about to read, so the size is capped
before it reaches the context, the content type is checked before
anything is parsed, and the result is handed over explicitly labelled
as untrusted - the same treatment web_search results already get.
"""
import re

from html import unescape
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests

from config import PAGE_FETCH_ENABLED, PAGE_MAX_CHARS, PAGE_TIMEOUT

# Tags whose contents are markup plumbing, not prose.
_SKIP = {"script", "style", "noscript", "template", "svg", "canvas",
         "head", "meta", "link", "iframe", "form", "button", "select"}

# Tags that should leave a line break behind them, so paragraphs don't
# run into each other and the model can see the structure.
_BREAK = {"p", "br", "div", "section", "article", "header", "footer",
          "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote",
          "pre", "figcaption", "td", "th"}

# Chrome's UA. Plenty of sites serve a stub or a 403 to anything that
# announces itself as a script, and getting an empty page back would
# look like a bug in here rather than a policy on their end.
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

_READABLE = ("text/html", "application/xhtml", "text/plain", "text/markdown")


class _Extractor(HTMLParser):
    """Collects visible text and the page title."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.title = ""
        self._depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._depth += 1
        elif tag == "title":
            self._in_title = True

        if tag in _BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP and self._depth:
            self._depth -= 1
        elif tag == "title":
            self._in_title = False

        if tag in _BREAK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._depth:
            self.parts.append(data)

    def text(self):
        joined = "".join(self.parts)
        # Collapse the acres of whitespace that HTML indentation leaves
        # behind, while keeping paragraph breaks.
        joined = re.sub(r"[ \t\r\f\v]+", " ", joined)
        joined = re.sub(r" *\n *", "\n", joined)
        joined = re.sub(r"\n{3,}", "\n\n", joined)

        return joined.strip()


def available():
    return PAGE_FETCH_ENABLED


def _tidy_url(url):
    url = str(url or "").strip()

    if not url:
        return None, "no address given"

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    parsed = urlparse(url)

    if not parsed.netloc or "." not in parsed.netloc:
        return None, f"{url!r} doesn't look like a web address"

    return url, ""


def fetch(url, limit=None):
    """Pull a page and return (text, detail) or (None, why_not)."""
    limit = limit or PAGE_MAX_CHARS
    url, problem = _tidy_url(url)

    if url is None:
        return None, problem

    try:
        response = requests.get(
            url,
            timeout=PAGE_TIMEOUT,
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html,text/*"},
            stream=True,
        )
    except requests.exceptions.RequestException as e:
        return None, f"couldn't reach it ({e})"

    try:
        if response.status_code != 200:
            return None, f"the site answered {response.status_code}"

        kind = (response.headers.get("Content-Type") or "").lower()

        if kind and not any(k in kind for k in _READABLE):
            return None, f"that's a {kind.split(';')[0]}, not a readable page"

        # Read with a ceiling rather than trusting Content-Length - it
        # can be absent, wrong, or enormous.
        raw = response.raw.read(limit * 8, decode_content=True) or b""
    except requests.exceptions.RequestException as e:
        return None, f"the download failed ({e})"
    finally:
        response.close()

    if not raw:
        return None, "the page was empty"

    body = raw.decode(response.encoding or "utf-8", errors="replace")

    if "html" in kind or body.lstrip()[:200].lower().startswith(("<!doctype", "<html")):
        parser = _Extractor()

        try:
            parser.feed(body)
        except Exception:
            # Malformed markup. Whatever was parsed before it broke is
            # still worth having.
            pass

        text, title = parser.text(), unescape(parser.title).strip()
    else:
        text, title = body.strip(), ""

    if not text:
        return None, "there was no readable text on it (it may need JavaScript)"

    truncated = len(text) > limit

    if truncated:
        text = text[:limit].rsplit(" ", 1)[0] + "..."

    detail = title or urlparse(url).netloc

    if truncated:
        detail += " (first part only)"

    return text, detail
