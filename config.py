import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LM_URL = "http://localhost:1234/v1/chat/completions"
SAMPLE_RATE = 16000

with open(os.path.join(BASE_DIR, "agent", "agent.json"), "r") as file:
    agent = json.load(file)

with open(os.path.join(BASE_DIR, "agent", "memory.json"), "r") as file:
    memory = json.load(file)

AGENT_NAME = agent["name"]
VOICE = agent.get("voice", "af_bella")  # falls back if agent.json has no "voice" key

_history_cfg = agent.get("history", {})
MAX_RAW_MESSAGES = _history_cfg.get("max_raw_messages", 15)
SUMMARIZE_CHUNK = _history_cfg.get("summarize_chunk", 8)

LONG_TERM_MEMORY_ENABLED = agent.get("long_term_memory", {}).get("enabled", True)
LONG_TERM_MEMORY_MAX_FACTS = agent.get("long_term_memory", {}).get("max_facts", 40)

_reminders_cfg = agent.get("reminders", {})
REMINDERS_ENABLED = _reminders_cfg.get("enabled", True)
REMINDER_CHECK_INTERVAL_SECONDS = _reminders_cfg.get("check_interval_minutes", 10) * 60

_web_search_cfg = agent.get("web_search", {})
WEB_SEARCH_ENABLED = _web_search_cfg.get("enabled", True)
WEB_SEARCH_MAX_RESULTS = _web_search_cfg.get("max_results", 5)
