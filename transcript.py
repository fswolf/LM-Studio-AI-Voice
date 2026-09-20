"""Everything that was ever said, which history.py doesn't keep.

history.py holds the last fifteen turns and folds the rest into a
running summary - then deletes them:

    _data["messages"] = messages[SUMMARIZE_CHUNK:]

That is the right call for the prompt, where context is the scarce
thing. It is the wrong call for the conversation. Ask her what you
decided about the deploy last Tuesday and the answer is gone, replaced
by two sentences of summary that were never written with that question
in mind.

So every turn is also appended here, to a file that summarization never
touches, and search_history reads it back. One line of JSON per turn,
appended and never rewritten, so a crash mid-write costs the last line
rather than the archive.

Search is deliberately plain: word overlap, recency as a tiebreak, no
embeddings and no index. The corpus is one person's conversations, the
file is a few megabytes after a year, and grepping it takes
milliseconds. A vector database here would be a way of making a simple
thing impressive rather than good.
"""
import json
import os
import re
import threading

from datetime import datetime

from config import BASE_DIR, TRANSCRIPT_ENABLED, TRANSCRIPT_MAX_MB

TRANSCRIPT_DIR = os.path.join(BASE_DIR, "history")
TRANSCRIPT_FILE = os.path.join(TRANSCRIPT_DIR, "transcript.jsonl")

_lock = threading.Lock()

# Fraction of the query's meaningful words a turn must contain. Low,
# because real questions share few words with the answer they're
# looking for - "my graphics card problems" matching "the AMD card" is
# one word in three, and is exactly the hit you wanted.
MIN_RELEVANCE = 0.3

_STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "but", "by", "can",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "him", "his", "how", "i", "if", "in", "is", "it", "its", "just", "know",
    "like", "me", "my", "not", "of", "on", "or", "our", "say", "said", "she",
    "so", "tell", "that", "the", "their", "them", "then", "there", "they",
    "this", "to", "told", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "you", "your",
    # Words that appear in any question *about* the archive and would
    # therefore match everything in it.
    "remember", "recall", "conversation", "talked", "talking", "discussed",
    "mentioned", "earlier", "before", "ago", "last", "time",
}


_STOP_STEMS = set()


def _stem(word):
    """Crudely chop a suffix so plurals and tenses match.

    "the cat reminder" should find "remind me to feed the cats", and
    without this it finds nothing at all - cats != cat, reminder !=
    remind. Not a real stemmer, and it doesn't need to be: both sides
    of the comparison go through the same function, so consistency
    matters and linguistic correctness doesn't.
    """
    for suffix in ("ing", "ies", "ers", "er", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]

    return word


def _words(text):
    return {
        _stem(word) for word in re.findall(r"[a-z0-9']+", str(text).lower())
        if len(word) > 2 and word not in _STOPWORDS
    } - _STOP_STEMS


_STOP_STEMS = {_stem(word) for word in _STOPWORDS}


def add(role, content):
    """Append one turn. Never raises - losing a line is survivable,
    losing the reply because the disk filled up is not."""
    if not TRANSCRIPT_ENABLED or not str(content or "").strip():
        return

    entry = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "role": role,
        "text": str(content).strip(),
    }

    try:
        with _lock:
            os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

            with open(TRANSCRIPT_FILE, "a+", encoding="utf-8") as handle:
                # A crash mid-write leaves a line with no newline on the
                # end. Appending straight onto it would glue this record
                # to the broken one and lose both, so the torn line gets
                # terminated first - it stays unparseable and gets
                # skipped on read, but it stops costing anything else.
                handle.seek(0, os.SEEK_END)

                if handle.tell():
                    handle.seek(handle.tell() - 1)

                    if handle.read(1) != "\n":
                        handle.write("\n")

                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except (OSError, ValueError, UnicodeDecodeError):
        pass


def _load():
    if not os.path.exists(TRANSCRIPT_FILE):
        return []

    entries = []

    try:
        with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()

                if not line:
                    continue

                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    # A torn line from a crash mid-write. Skip it; the
                    # rest of the archive is still perfectly good.
                    continue
    except OSError:
        return []

    return entries


def size_mb():
    try:
        return os.path.getsize(TRANSCRIPT_FILE) / (1024 * 1024)
    except OSError:
        return 0.0


def count():
    return len(_load())


def search(query, limit=5, window=1):
    """Find past turns matching a description.

    `window` pulls in the turns either side of each hit, because half a
    conversation is usually the part that answers the question - a bare
    "yeah, Friday works" means nothing without what came before it.
    """
    terms = _words(query)
    entries = _load()

    if not entries:
        return []

    if not terms:
        return entries[-limit:]

    scored = []

    for index, entry in enumerate(entries):
        overlap = terms & _words(entry.get("text", ""))

        if not overlap:
            continue

        # Proportion of the question answered, not raw hit count, so a
        # long rambling turn doesn't beat a short exact one. Recency
        # only breaks ties.
        score = len(overlap) / len(terms)

        # Word overlap has no idea what words mean, so a single
        # incidental match ("something we never discussed" finding
        # "something boring and obvious") is noise. A floor cuts the
        # worst of it; the model is told to judge the rest, because a
        # lexical search genuinely cannot tell relevance from
        # coincidence and shouldn't pretend to.
        if score < MIN_RELEVANCE:
            continue

        scored.append((score + (index / len(entries)) * 0.05, index))

    if not scored:
        return []

    scored.sort(reverse=True)

    keep = set()

    for _score, index in scored[:limit]:
        for near in range(index - window, index + window + 1):
            if 0 <= near < len(entries):
                keep.add(near)

    return [entries[i] for i in sorted(keep)]


def describe(entries, agent_name="assistant"):
    """Format hits for the model, grouped by day."""
    if not entries:
        return "Nothing in the transcript matches that."

    lines = []
    day = None

    for entry in entries:
        try:
            when = datetime.fromisoformat(entry["at"])
        except (KeyError, ValueError):
            continue

        if when.date() != day:
            day = when.date()
            lines.append(f"\n{when.strftime('%A %d %B %Y')}")

        who = "You" if entry.get("role") == "user" else agent_name
        text = entry.get("text", "").replace("\n", " ")

        if len(text) > 300:
            text = text[:300].rsplit(" ", 1)[0] + "..."

        lines.append(f"  {when.strftime('%H:%M')} {who}: {text}")

    return "\n".join(lines).strip()


def trim():
    """Drop the oldest half once the file passes its cap.

    Called at startup rather than on every write: the archive is meant
    to be long, and rewriting a multi-megabyte file mid-conversation to
    save a few kilobytes is a bad trade.
    """
    if not TRANSCRIPT_ENABLED or size_mb() <= TRANSCRIPT_MAX_MB:
        return False

    entries = _load()

    if len(entries) < 2:
        return False

    keep = entries[len(entries) // 2:]

    try:
        with _lock:
            temporary = TRANSCRIPT_FILE + ".new"

            with open(temporary, "w", encoding="utf-8") as handle:
                for entry in keep:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

            os.replace(temporary, TRANSCRIPT_FILE)
    except OSError:
        return False

    return True
