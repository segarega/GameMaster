# Copyright (C) 2026 SEGAREGA
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
GameMaster - OpenAI-compatible proxy with GM filtering

Usage:
    python main.py

Endpoints:
    POST /v1/chat/completions - Chat with GM filtering
    GET  /health              - Health check
    GET  /stats               - Service stats
    POST /reindex             - Force reindex
    GET  /debug/sections      - Debug indexed sections
"""

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
import httpx

from config.settings import Settings
from gm.retriever import GMContentManager
from gm.prompt_filter import PromptFilter
from gm.selector import GMSelectorClient
from character_memory import CharacterMemoryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
settings: Settings = None
retriever: GMContentManager = None
prompt_filter: PromptFilter = None
selector_client: GMSelectorClient = None
character_memory_manager: CharacterMemoryManager = None
http_client: httpx.AsyncClient = None
start_time: float = 0

# Protects runtime component swaps during hot-reload. The game normally sends one
# request at a time, and serializing reload/request handling prevents a reload
# from closing a selector client while a request is still using it.
runtime_lock: asyncio.Lock = asyncio.Lock()


DEFAULT_REQUEST_TYPE_SIGNATURES: Dict[str, List[str]] = {
    "dialogue": [
        "### Mission ###\nRole-play as a character in Mount & Blade II: Bannerlord. Use your personality, history, and context to inform responses. Output ONLY a valid JSON object with no extra text or markdown.",
        "===== GROUP CONVERSATION MODE =====",
        "===== NPC-TO-NPC CONVERSATION MODE =====",
    ],
    "events": [
        "## EVENT STRUCTURE:\nMUST include: 1) CAUSE (from data) 2) ACTION (decision taken) 3) CONSEQUENCE (future impact)\nPrefer DEVELOPING existing conflicts over new minor incidents. Return [] if insufficient data."
    ],
    "diplomacy": [
        "### CRITICAL REMINDER: You Are a Living Ruler ###"
    ],
}
DEFAULT_REQUEST_TYPE_PRIORITY: List[str] = ["diplomacy", "events", "dialogue"]
NPC_TO_NPC_CONVERSATION_MARKER = "===== NPC-TO-NPC CONVERSATION MODE ====="
GROUP_CONVERSATION_MARKER = "===== GROUP CONVERSATION MODE ====="


def _json_for_log(value: Any) -> str:
    """Serialize JSON for the optional upstream request/response log."""
    try:
        if settings and settings.llm_log_pretty_json:
            return json.dumps(value, ensure_ascii=False, indent=2)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(value)


def _raw_content_for_log(content: Any) -> str:
    """Render raw log content, prettifying JSON strings when requested."""
    text = content if isinstance(content, str) else str(content)
    if settings and settings.llm_log_pretty_json:
        try:
            parsed = json.loads(text)
            return _json_for_log(parsed)
        except Exception:
            pass
    return text


def _log_block_text(label: str, request_id: str, content: str) -> str:
    timestamp = datetime.now(timezone.utc).isoformat()
    return (
        f"\n\n===== {label} | request_id={request_id} | utc={timestamp} =====\n"
        f"{content}\n"
        f"===== END {label} | request_id={request_id} =====\n"
    )


def _append_text_file(path: str, text: str) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text)


async def append_llm_log(label: str, request_id: str, content: Any, *, raw: bool = False) -> None:
    """Append an upstream LLM request/response log entry if enabled."""
    if not settings or not settings.llm_log_enabled:
        return
    try:
        text = _raw_content_for_log(content) if raw else _json_for_log(content)
        await asyncio.to_thread(
            _append_text_file,
            settings.llm_log_path,
            _log_block_text(label, request_id, text),
        )
    except Exception as e:
        logger.warning(f"Failed to write LLM request/response log: {e}")


# Pydantic models for OpenAI compatibility
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: str | List[str] | None = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None
    # Extra fields for GM filtering
    use_gm: Optional[bool] = None
    # Optional hint from caller. If absent or conflicting, the proxy detects the type from prompt signatures.
    request_type: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class CharacterMemoryActionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    create_backup: Optional[bool] = False


def _normalize_request_type(raw: Optional[str]) -> Optional[str]:
    """Normalize request type hints to the three AIInfluence modes."""
    if not raw:
        return None
    value = str(raw).strip().lower().replace('-', '_').replace(' ', '_')
    if value in {"dialogue", "dialog", "chat"}:
        return "dialogue"
    if "event" in value:
        return "events"
    if "diplom" in value or "statement" in value or "kingdom" in value:
        return "diplomacy"
    return None


def _message_content_to_text(content: Any) -> str:
    """Convert OpenAI message content to text for prompt signature detection."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _is_text_only_content_part(item: Any) -> bool:
    """Return True when a structured content item is purely textual."""
    if isinstance(item, str):
        return True
    if not isinstance(item, dict):
        return False
    item_type = str(item.get("type", "") or "").strip().lower()
    if item_type not in {"", "text"}:
        return False
    return isinstance(item.get("text"), str) or isinstance(item.get("content"), str)


def _build_filtered_content_preserving_shape(original_content: Any, filtered_text: str) -> Any | None:
    """Rewrite filtered content without dropping non-text parts."""
    if isinstance(original_content, str) or original_content is None:
        return filtered_text
    if isinstance(original_content, list) and all(_is_text_only_content_part(item) for item in original_content):
        return [{"type": "text", "text": filtered_text}]
    return None


def _add_repair_candidate(candidates: List[tuple[str, str]], text: str, description: str) -> None:
    """Add a unique JSON-repair candidate."""
    if not isinstance(text, str):
        return
    cleaned = text.strip()
    if not cleaned:
        return
    for existing_text, _ in candidates:
        if existing_text == cleaned:
            return
    candidates.append((cleaned, description))


def _strip_code_fences(text: str) -> str:
    """Remove a single outer Markdown code-fence wrapper."""
    cleaned = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*", "", text.strip())
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    return cleaned.strip()


def _remove_trailing_commas(text: str) -> str:
    """Remove trailing commas before closing JSON brackets/braces."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def _looks_like_valid_json_value(value: str) -> bool:
    """Heuristic check for an already-valid bare JSON value."""
    if not value or not value.strip():
        return False
    value = value.strip()
    if value.startswith('"') or value.startswith("{") or value.startswith("["):
        return True
    if re.match(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$", value):
        return True
    return value.lower() in {"true", "false", "null"}


def _quote_malformed_bare_values(text: str) -> str:
    """Quote obvious malformed bare JSON values after a field name."""
    return re.sub(
        r'("(?P<field>[A-Za-z0-9_]+)"\s*:\s*)(?P<value>[^,}\]]+)',
        lambda match: (
            match.group(0)
            if _looks_like_valid_json_value(match.group("value").strip())
            else match.group(1) + json.dumps(match.group("value").strip(), ensure_ascii=False)
        ),
        text,
    )


def _extract_balanced_json(text: str, start_char: str, end_char: str) -> str | None:
    """Extract the first balanced JSON object/array substring from text."""
    start_index = text.find(start_char)
    if start_index < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start_index, len(text)):
        char = text[i]
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                return text[start_index:i + 1]
    return None


def _extract_json_substring(text: str) -> str | None:
    """Extract the first balanced JSON object or array from surrounding text."""
    return _extract_balanced_json(text, "{", "}") or _extract_balanced_json(text, "[", "]")


def _escape_string_newlines(text: str) -> str:
    """Escape raw CR/LF characters that occur inside JSON strings."""
    builder: List[str] = []
    in_string = False
    escaped = False
    for character in text:
        if escaped:
            builder.append(character)
            escaped = False
            continue
        if character == "\\" and in_string:
            builder.append(character)
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
            builder.append(character)
            continue
        if in_string and character == "\n":
            builder.append("\\n")
            continue
        if in_string and character == "\r":
            builder.append("\\r")
            continue
        builder.append(character)
    return "".join(builder)


def _try_normalize_json(text: str) -> tuple[bool, str]:
    """Parse and normalize JSON into compact form."""
    try:
        token = json.loads(text)
    except Exception:
        return False, ""
    return True, json.dumps(token, ensure_ascii=False, separators=(",", ":"))


def repair_json_content(content: str) -> tuple[str, bool, str]:
    """Attempt conservative repairs on malformed JSON-like assistant content.

    This helper must never mutate assistant content unless the repaired result
    parses cleanly as JSON. A cosmetic change that still leaves invalid JSON is
    worse than no change at all because it can destroy provider output that a
    downstream layer might otherwise recover or inspect.
    """
    if not isinstance(content, str) or not content.strip():
        return content, False, ""

    original = content
    trimmed = content.strip()
    without_fences = _strip_code_fences(trimmed)

    candidates: List[tuple[str, str]] = []
    _add_repair_candidate(candidates, without_fences, "trimmed whitespace / stripped markdown code fences")
    _add_repair_candidate(candidates, _remove_trailing_commas(without_fences), "removed trailing commas")
    _add_repair_candidate(
        candidates,
        _quote_malformed_bare_values(_remove_trailing_commas(without_fences)),
        "quoted malformed bare JSON values",
    )

    json_substring = _extract_json_substring(without_fences)
    if json_substring and json_substring.strip():
        _add_repair_candidate(candidates, json_substring, "extracted JSON object/array from surrounding text")
        _add_repair_candidate(
            candidates,
            _remove_trailing_commas(json_substring),
            "extracted JSON object/array and removed trailing commas",
        )
        _add_repair_candidate(
            candidates,
            _quote_malformed_bare_values(_remove_trailing_commas(json_substring)),
            "extracted JSON object/array and quoted malformed bare JSON values",
        )

    escaped_newlines = _escape_string_newlines(_remove_trailing_commas(without_fences))
    _add_repair_candidate(candidates, escaped_newlines, "escaped literal newlines inside JSON strings")
    _add_repair_candidate(
        candidates,
        _quote_malformed_bare_values(escaped_newlines),
        "escaped literal string newlines and quoted malformed bare JSON values",
    )

    if json_substring and json_substring.strip():
        escaped_substring = _escape_string_newlines(_remove_trailing_commas(json_substring))
        _add_repair_candidate(
            candidates,
            escaped_substring,
            "extracted JSON object/array and escaped literal string newlines",
        )
        _add_repair_candidate(
            candidates,
            _quote_malformed_bare_values(escaped_substring),
            "extracted JSON object/array, escaped literal string newlines, and quoted malformed bare JSON values",
        )

    for candidate_text, description in candidates:
        parsed, normalized = _try_normalize_json(candidate_text)
        if parsed:
            was_changed = normalized != original.strip()
            summary = (
                f"LLM content JSON repaired: {description}."
                if was_changed
                else "LLM content JSON was already valid."
            )
            return normalized, was_changed, summary

    return original, False, "LLM content was not parseable as JSON; left unchanged."


def _repair_chat_completion_content(result: Dict[str, Any]) -> List[str]:
    """Repair assistant message.content fields in a non-stream chat completion response."""
    summaries: List[str] = []
    choices = result.get("choices")
    if not isinstance(choices, list):
        return summaries

    for idx, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        repaired, was_changed, summary = repair_json_content(content)
        if was_changed:
            message["content"] = repaired
        if summary:
            summaries.append(f"choice {idx}: {summary}")

    return summaries


def _combined_prompt_text(messages: List[Dict[str, Any]]) -> str:
    return "\n".join(_message_content_to_text(msg.get("content")) for msg in messages)


def _looks_like_filterable_dialogue_system_prompt(text: str) -> bool:
    """Heuristic for the AIInfluence dialogue prompt-container system message.

    Preset/model-routing system prompts may precede the real AIInfluence prompt. We should
    not try to filter those. Only target system messages that look like the large injected
    dialogue container or explicit policy-tagged lore blocks.
    """
    if not isinstance(text, str):
        return False
    haystack = text.strip()
    if not haystack:
        return False

    strong_markers = (
        "### Mission ###",
        "### Character Briefing (CURRENT DATA) ###",
        "### Immediate Situation (CURRENT DATA) ###",
        "### The Player (CURRENT DATA) ###",
        "### Conversation History ###",
        "### The World ###",
        "### Mentioned Settlements ###",
        "### Nearby Settlements (Strategic Context, CURRENT DATA) ###",
        "### Nearby Parties (NPC Vicinity, CURRENT DATA) ###",
        "### Global Politics of the World ###",
        "=== [GM",
        "=== [PINNED",
        "=== [IGNORE",
    )
    if any(marker in haystack for marker in strong_markers):
        return True

    # Fallback: large system prompts that contain multiple AIInfluence-style section headers.
    ai_heading_count = len(re.findall(r'(?m)^\s*(?:#{3,6}\s+.+?|={3,}\s*[^=\s].+?={3,})\s*$', haystack))
    return len(haystack) >= 4000 and ai_heading_count >= 3


def _has_prompt_policy_markers(text: str) -> bool:
    """Return True when a message contains explicit GM/PINNED/IGNORE policy tags."""
    if not isinstance(text, str):
        return False
    return bool(re.search(
        r'\[(?:PINNED|PIN|GM|IGNORE|END\s+(?:PINNED|PIN|GM|IGNORE))(?::[A-Z0-9_ -]+)*\]',
        text,
        re.IGNORECASE,
    ))


def _approx_prompt_tokens(messages: List[Dict[str, Any]]) -> int:
    """Rough local estimate of prompt tokens for diagnostics."""
    chars = sum(len(_message_content_to_text(msg.get("content"))) for msg in messages)
    return chars // 4


def _contains_npc_to_npc_conversation_marker(messages: List[Dict[str, Any]]) -> bool:
    """Return True when the intercepted request is AIInfluence NPC-to-NPC mode."""
    return any(
        NPC_TO_NPC_CONVERSATION_MARKER in _message_content_to_text(msg.get("content"))
        for msg in messages
    )


def _contains_group_conversation_marker(messages: List[Dict[str, Any]]) -> bool:
    """Return True when the intercepted request is AIInfluence group conversation mode."""
    return any(
        GROUP_CONVERSATION_MARKER in _message_content_to_text(msg.get("content"))
        for msg in messages
    )


def _strip_user_messages_for_special_dialogue_modes_if_enabled(
    messages: List[Dict[str, Any]],
    npc_to_npc_marker_present: bool,
    group_marker_present: bool,
) -> List[Dict[str, Any]]:
    """Optionally remove user-role messages from special AIInfluence dialogue modes."""
    if not settings:
        return messages

    active_modes: List[str] = []
    if (
        bool(getattr(settings, "disable_user_last_message_during_npc_npc_conversation", False))
        and npc_to_npc_marker_present
    ):
        active_modes.append("NPC-to-NPC conversation")
    if (
        bool(getattr(settings, "disable_user_last_message_during_group_chat", False))
        and group_marker_present
    ):
        active_modes.append("group chat")

    if not active_modes:
        return messages

    stripped = [msg for msg in messages if msg.get("role") != "user"]
    removed = len(messages) - len(stripped)
    if removed:
        logger.info(
            "%s mode detected; removed %d user message(s) from outbound request",
            " and ".join(active_modes),
            removed,
        )
    return stripped


def _summarize_messages_for_log(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compact per-message diagnostics for the outbound payload."""
    summary: List[Dict[str, Any]] = []
    for i, msg in enumerate(messages):
        content = msg.get("content")
        text = _message_content_to_text(content)
        entry: Dict[str, Any] = {
            "index": i,
            "role": msg.get("role"),
            "content_chars": len(text),
            "content_tokens_est": len(text) // 4,
            "content_type": type(content).__name__,
        }
        if isinstance(content, list):
            entry["parts"] = len(content)
        summary.append(entry)
    return summary


def _serialize_backend_request(backend_request: Dict[str, Any]) -> tuple[str, int, str]:
    """Serialize the outbound request for wire-size/hash diagnostics."""
    wire_json = json.dumps(backend_request, ensure_ascii=False, separators=(",", ":"))
    wire_chars = len(wire_json)
    wire_sha256 = hashlib.sha256(wire_json.encode("utf-8")).hexdigest()
    return wire_json, wire_chars, wire_sha256


def _extract_provider_prompt_tokens(result: Any) -> int | None:
    """Extract provider-reported prompt token count from a response payload."""
    if not isinstance(result, dict):
        return None
    usage = result.get("usage")
    pricing = result.get("x_nanogpt_pricing")
    if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
        return usage.get("prompt_tokens")
    if isinstance(pricing, dict) and isinstance(pricing.get("inputTokens"), int):
        return pricing.get("inputTokens")
    return None


def _normalize_signature_text(text: str) -> str:
    """Normalize newlines/spacing enough for stable multi-line signature matching."""
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    # Keep punctuation/case mostly intact, but collapse horizontal whitespace around newlines.
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(lines).strip().lower()


def detect_request_type_from_prompt(messages: List[Dict[str, Any]], explicit_request_type: Optional[str]) -> str:
    """Resolve dialogue/events/diplomacy from configured prompt signatures.

    Prompt signatures win over explicit request_type because AIInfluence/other clients may omit
    or mislabel the extra field. If no signature and no valid hint exists, return "unknown"
    so scoped tags like [PINNED:DIALOGUE] do not accidentally match.
    """
    explicit = _normalize_request_type(explicit_request_type)
    prompt_text = _normalize_signature_text(_combined_prompt_text(messages))

    matches: List[str] = []
    signatures = settings.request_type_signatures if settings else DEFAULT_REQUEST_TYPE_SIGNATURES
    for req_type in DEFAULT_REQUEST_TYPE_PRIORITY:
        for sig in signatures.get(req_type, []):
            sig_norm = _normalize_signature_text(sig)
            if sig_norm and sig_norm in prompt_text:
                matches.append(req_type)
                break

    if matches:
        resolved = matches[0]
        if explicit and explicit != resolved:
            logger.warning(
                "request_type hint %r conflicts with prompt signature %r; using prompt signature",
                explicit, resolved,
            )
        return resolved

    if explicit:
        logger.info("No request-type signature matched; using explicit request_type=%s", explicit)
        return explicit

    logger.warning("No request-type signature matched and no valid request_type provided; using request_type=unknown")
    return "unknown"


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global settings, retriever, prompt_filter, selector_client, character_memory_manager, http_client, start_time
    
    logger.info("Starting GameMaster service...")
    start_time = time.time()
    
    # Load settings
    settings = Settings.load()
    logger.info(f"Loaded settings from: {settings.config_path or 'defaults'}")
    logger.info(f"Backend: {settings.get_chat_completions_url()}")
    
    logger.info("Filtering mode: %s", settings.get_filtering_mode())

    # Initialize optional selector model.
    selector_client = None
    if settings.uses_selector_filtering():
        selector_client = GMSelectorClient(settings)
        await selector_client.initialize()
        if not selector_client.enabled:
            raise RuntimeError(
                "Selector mode is configured, but selector settings are incomplete. "
                "Set selector.api_url and selector.model."
            )

    # Initialize retriever
    retriever = GMContentManager(settings)
    await retriever.initialize()
    logger.info(f"GM content manager initialized: {retriever.get_stats()}")
    
    # Initialize prompt filter
    prompt_filter = PromptFilter(retriever, settings, selector_client=selector_client)

    # Initialize Character Memory Control. Auto mode is manual settings driven and
    # never indexes/reindexes Static GM content.
    character_memory_manager = CharacterMemoryManager(settings)
    await character_memory_manager.start_auto()
    
    # Initialize HTTP client for backend
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0)  # 5 minute timeout for long completions
    )
    
    logger.info("GameMaster service ready!")
    
    yield
    
    # Cleanup
    logger.info("Shutting down GameMaster service...")
    if http_client:
        await http_client.aclose()
    if character_memory_manager:
        await character_memory_manager.stop_auto()
    if retriever:
        await retriever.aclose()
    if selector_client:
        await selector_client.aclose()
    logger.info("Cleanup complete")


# Create FastAPI app
app = FastAPI(
    title="GameMaster",
    description="OpenAI-compatible proxy with selector filtering for AIInfluence",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "GameMaster"}


@app.get("/stats")
async def stats():
    """Get service statistics."""
    return {
        "gm_content_stats": retriever.get_stats() if retriever else {},
        "selector_stats": selector_client.get_stats() if selector_client else {},
        "settings": {
            "config_path": settings.config_path if settings else None,
            "backend": settings.get_chat_completions_url() if settings else None,
            "models": settings.models if settings else {},
            "gm_filtering": {
                "max_event_history": settings.max_event_history,
                "dynamic_filter_enabled": settings.dynamic_filter_enabled,
                "disable_user_last_message_during_npc_npc_conversation": settings.disable_user_last_message_during_npc_npc_conversation,
                "disable_user_last_message_during_group_chat": settings.disable_user_last_message_during_group_chat,
                "fuzzy_match_threshold": settings.fuzzy_match_threshold,
                "max_people_present": settings.max_people_present,
                "max_nearby_settlements": settings.max_nearby_settlements,
                "max_nearby_parties": settings.max_nearby_parties,
                "max_inventory_lines": settings.max_inventory_lines,
            } if settings else {},
            "request_parameters": settings.request_parameters if settings else {},
            "system_prompts": {
                key: {
                    "pre_history_chars": len(value.get("pre_history", "") or ""),
                    "post_history_chars": len(value.get("post_history", "") or ""),
                } for key, value in (settings.system_prompts or {}).items()
            } if settings else {},
            "llm_logging": {
                "enabled": settings.llm_log_enabled,
                "path": settings.llm_log_path,
                "pretty_json": settings.llm_log_pretty_json,
            } if settings else {},
            "selector": {
                "enabled": settings.selector_enabled,
                "api_url": settings.selector_api_url,
                "model": settings.selector_model,
                "temperature": settings.selector_temperature,
                "max_tokens": settings.selector_max_tokens,
                "timeout_seconds": settings.selector_timeout_seconds,
                "context_rules": settings.selector_context_rules,
                "log_enabled": settings.selector_log_enabled,
                "log_path": settings.selector_log_path,
            } if settings else {},
            "filtering": {
                "mode": settings.get_filtering_mode(),
            } if settings else {},
            "static_gm_index": {
                "enabled": settings.static_gm_index_enabled,
                "ai_influence_folder": settings.static_gm_index_ai_influence_folder,
                "files": settings.static_gm_index_files,
                "db_path": settings.static_gm_index_db_path,
                "manual_reindex_only": True,
                "selector_payload": settings.static_gm_index_selector_payload,
                "summary_enabled": settings.static_gm_index_summary_enabled,
                "summary_model": settings.static_gm_index_summary_model,
                "effective_summary_model": (
                    settings.static_gm_index_summary_model
                    or settings.selector_model
                    or next((m for m in settings.models.values() if m), "")
                ),
                "summary_prompt_chars": len(settings.static_gm_index_summary_instruction or ""),
            } if settings else {},
            "character_memory": {
                "enabled": settings.character_memory_enabled,
                "campaign_dir": settings.character_memory_campaign_dir,
                "auto_enabled": settings.character_memory_auto_enabled,
                "preserve_last_lines": settings.character_memory_preserve_last_lines,
                "auto_trigger_raw_lines": settings.character_memory_auto_trigger_raw_lines,
                "model": settings.character_memory_model,
                "effective_model": (
                    settings.character_memory_model
                    or settings.selector_model
                    or next((m for m in settings.models.values() if m), "")
                ),
                "summary_prompt_chars": len(settings.character_memory_summary_prompt or ""),
                "merge_prompt_chars": len(settings.character_memory_merge_prompt or ""),
                "profile_prompt_chars": len(settings.character_memory_profile_update_prompt or ""),
                "runtime": character_memory_manager.get_status() if character_memory_manager else {},
            } if settings else {},
            "request_type_detection": {
                "signatures": settings.request_type_signatures,
            } if settings else {},
        },
        "uptime_seconds": time.time() - start_time
    }


async def _reload_runtime_locked(config_path_override: Optional[str] = None) -> Dict[str, Any]:
    """Reload settings and rebuild runtime objects without reindexing content.

    Caller must hold runtime_lock. Rebuilding objects is required because the
    selector client, prompt filter, and StaticGMIndex copy settings at construction
    time. StaticGMIndex only opens the existing SQLite DB here; summary generation
    and content indexing happen exclusively via POST /reindex / the GUI button.
    """
    global settings, retriever, prompt_filter, selector_client, character_memory_manager

    config_path = str(config_path_override or "").strip() or (settings.config_path if settings and settings.config_path else None)
    logger.info("Hot-reloading GameMaster settings from %s", config_path or "default search paths")

    old_selector_client = selector_client
    old_retriever = retriever
    old_character_memory_manager = character_memory_manager
    new_selector_client: GMSelectorClient | None = None
    new_retriever: GMContentManager | None = None
    new_character_memory_manager: CharacterMemoryManager | None = None
    try:
        # GUI hot reload supplies an explicit config_path. In that case the GUI-edited
        # settings.json must be authoritative; stale GMR_SELECTOR_* / GMR_MODEL_*
        # environment variables should not silently override selector model/temperature.
        new_settings = Settings.load(config_path, apply_env_overrides=not bool(config_path_override))

        if new_settings.uses_selector_filtering():
            new_selector_client = GMSelectorClient(new_settings)
            await new_selector_client.initialize()
            if not new_selector_client.enabled:
                await new_selector_client.aclose()
                raise RuntimeError(
                    "Selector mode is configured, but selector settings are incomplete. "
                    "Set selector.api_url and selector.model."
                )

        new_retriever = GMContentManager(new_settings)
        await new_retriever.initialize()
        new_prompt_filter = PromptFilter(
            new_retriever,
            new_settings,
            selector_client=new_selector_client,
        )
        new_character_memory_manager = CharacterMemoryManager(new_settings)
        await new_character_memory_manager.start_auto()

        settings = new_settings
        selector_client = new_selector_client
        retriever = new_retriever
        prompt_filter = new_prompt_filter
        character_memory_manager = new_character_memory_manager

        if old_selector_client and old_selector_client is not new_selector_client:
            try:
                await old_selector_client.aclose()
            except Exception as close_error:
                logger.warning("Old selector client did not close cleanly after reload: %s", close_error)
        if old_retriever and old_retriever is not new_retriever:
            try:
                await old_retriever.aclose()
            except Exception as close_error:
                logger.warning("Old retriever did not close cleanly after reload: %s", close_error)
        if old_character_memory_manager and old_character_memory_manager is not new_character_memory_manager:
            try:
                await old_character_memory_manager.stop_auto()
            except Exception as close_error:
                logger.warning("Old Character Memory manager did not stop cleanly after reload: %s", close_error)

        static_summary_model = getattr(settings, "static_gm_index_summary_model", "") or "<selector model>"
        logger.info(
            "Hot-reload complete: backend=%s mode=%s selector_model=%s summary_model=%s summary_prompt_chars=%s",
            settings.get_chat_completions_url(),
            settings.get_filtering_mode(),
            settings.selector_model,
            static_summary_model,
            len(getattr(settings, "static_gm_index_summary_instruction", "") or ""),
        )
        return {
            "status": "success",
            "message": "Settings reloaded successfully",
            "config_path": settings.config_path,
            "filtering_mode": settings.get_filtering_mode(),
            "selector_model": settings.selector_model,
            "selector_temperature": settings.selector_temperature,
            "selector_max_tokens": settings.selector_max_tokens,
            "static_gm_index_summary_model": settings.static_gm_index_summary_model,
            "static_gm_index_summary_prompt_chars": len(settings.static_gm_index_summary_instruction or ""),
            "backend": settings.get_chat_completions_url(),
            "character_memory_model": settings.character_memory_model,
            "character_memory_auto_enabled": settings.character_memory_auto_enabled,
        }
    except Exception as e:
        if new_selector_client and new_selector_client is not old_selector_client:
            try:
                await new_selector_client.aclose()
            except Exception:
                pass
        if new_retriever and new_retriever is not old_retriever:
            try:
                await new_retriever.aclose()
            except Exception:
                pass
        if new_character_memory_manager and new_character_memory_manager is not old_character_memory_manager:
            try:
                await new_character_memory_manager.stop_auto()
            except Exception:
                pass
        logger.error("Failed to hot-reload settings: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to hot-reload settings: {e}")


async def _request_config_path(request: Request) -> Optional[str]:
    """Return optional config path supplied by the GUI for hot reload/reindex."""
    header_path = str(request.headers.get("X-GameMaster-Config-Path", "") or "").strip()
    if header_path:
        return header_path
    try:
        payload = await request.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        value = str(payload.get("config_path", "") or "").strip()
        if value:
            return value
    return None


@app.post("/reload")
async def reload_settings(request: Request):
    """Reload settings and rebuild runtime components from disk."""
    config_path = await _request_config_path(request)
    async with runtime_lock:
        return await _reload_runtime_locked(config_path)


def _configured_gm_enabled(request: ChatCompletionRequest) -> bool:
    """Resolve whether GM filtering is enabled for this request."""
    if request.use_gm is not None:
        return bool(request.use_gm)
    return True


def _inject_configured_system_prompts(
    messages: List[Dict[str, Any]],
    resolved_request_type: str,
) -> List[Dict[str, Any]]:
    """Insert optional pre/post system prompts after filtering.

    Injecting after request-type detection and GM filtering keeps these extra system prompts
    from being mistaken for the large AIInfluence prompt container. The pre-history prompt is
    the first system prompt in the outbound list, and post-history is the final outbound system
    prompt.
    """
    if not settings:
        return messages

    pre_history, post_history = settings.get_system_prompt_pair(resolved_request_type)
    outbound = list(messages)
    if pre_history.strip():
        outbound.insert(0, {"role": "system", "content": pre_history.strip()})
    if post_history.strip():
        outbound.append({"role": "system", "content": post_history.strip()})
    return outbound


def _build_backend_request(
    request: ChatCompletionRequest,
    model: str,
    messages_to_send: List[Dict[str, Any]],
    resolved_request_type: str,
) -> Dict[str, Any]:
    """Build outbound JSON while omitting unchecked request parameters.

    temperature/top_p/top_k are only sent when enabled in settings.json/GUI for the
    resolved request type. Request-provided values for those fields are deliberately
    ignored so an unchecked GUI box really means "send nothing".
    """
    backend_request: Dict[str, Any] = {}

    # Preserve provider/model-specific extras such as tools, response_format, seed, route, etc.
    # Internal proxy controls and sampling params handled by settings are excluded.
    excluded = {
        "model", "messages", "stream", "temperature", "top_p", "top_k",
        "use_gm", "request_type",
    }
    for key, value in (getattr(request, "model_extra", None) or {}).items():
        if key not in excluded and value is not None:
            backend_request[key] = value

    backend_request.update({
        "model": model,
        "messages": messages_to_send,
        # This proxy still returns a normal non-streaming response, so do not advertise streaming upstream.
        "stream": False,
    })

    # Pass through standard non-sampling request fields only if the client explicitly supplied them.
    fields_set = set(getattr(request, "model_fields_set", set()) or set())
    for field_name in ("max_tokens", "stop", "presence_penalty", "frequency_penalty", "user"):
        if field_name in fields_set:
            value = getattr(request, field_name, None)
            if value is not None:
                backend_request[field_name] = value

    if settings:
        backend_request.update(settings.get_enabled_request_parameters(resolved_request_type))

    return backend_request


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible /v1/chat/completions endpoint with GM filtering."""
    async with runtime_lock:
        return await _chat_completions_impl(request)


async def _chat_completions_impl(request: ChatCompletionRequest):
    """
    OpenAI-compatible /v1/chat/completions endpoint with GM filtering.
    
    Additional fields:
    - use_gm: bool - Enable/disable GM filtering (default: true)
    - request_type: str - Type of request for optimized filtering (dialogue, events, diplomacy)
    """
    request_start = time.time()
    request_id = uuid.uuid4().hex
    
    try:
        # Convert messages to dict format
        messages_dict = [msg.model_dump(exclude_none=True) for msg in request.messages]
        npc_to_npc_marker_present = _contains_npc_to_npc_conversation_marker(messages_dict)
        group_marker_present = _contains_group_conversation_marker(messages_dict)
        resolved_request_type = detect_request_type_from_prompt(messages_dict, request.request_type)
        logger.info(
            "Resolved request_type=%s (hint=%s)",
            resolved_request_type,
            request.request_type or "none",
        )
        gm_filter_enabled = _configured_gm_enabled(request)
        
        # Apply GM filtering if enabled.
        # Dialogue: use the last user line as the selector query and filter each system message
        # in place. Untagged content stays; explicit [GM]/[PINNED]/[IGNORE] drive the changes.
        # Events/diplomacy: filter whichever message(s) actually contain explicit policy tags.
        # Some providers place the huge AIInfluence world/rules payload in the system message,
        # while the game-state task lives in the user message.
        if gm_filter_enabled and prompt_filter:
            logger.info(f"Applying GM filtering for {resolved_request_type} request")
            messages_to_send = [dict(msg) for msg in messages_dict]
            stats_rollup = {
                "original_size": 0,
                "filtered_size": 0,
                "sections_included": 0,
                "pinned_sections_included": 0,
            }
            target_roles: List[str] = []
            filtered_target_indices: List[int] = []

            if resolved_request_type == "dialogue":
                system_candidates = [
                    i for i, msg in enumerate(messages_to_send)
                    if msg.get("role") == "system" and msg.get("content") is not None
                ]
                target_indices = [
                    i for i in system_candidates
                    if _looks_like_filterable_dialogue_system_prompt(
                        _message_content_to_text(messages_to_send[i].get("content"))
                    )
                ]
                if not target_indices and system_candidates:
                    target_indices = [system_candidates[-1]]
                    logger.warning(
                        "Dialogue request had system prompts but none matched AIInfluence markers; "
                        "falling back to filtering only the last system message at index %s",
                        system_candidates[-1],
                    )
            else:
                content_indices = [
                    i for i, msg in enumerate(messages_to_send)
                    if msg.get("role") in {"system", "user"} and msg.get("content") is not None
                ]
                policy_tagged_indices = [
                    i for i in content_indices
                    if _has_prompt_policy_markers(
                        _message_content_to_text(messages_to_send[i].get("content"))
                    )
                ]
                if policy_tagged_indices:
                    target_indices = policy_tagged_indices
                else:
                    user_indices = [
                        i for i in content_indices
                        if messages_to_send[i].get("role") == "user"
                    ]
                    system_indices = [
                        i for i in content_indices
                        if messages_to_send[i].get("role") == "system"
                    ]
                    if user_indices:
                        target_indices = [user_indices[-1]]
                    elif system_indices:
                        target_indices = [system_indices[-1]]
                    else:
                        target_indices = [0] if messages_to_send else []
                    logger.warning(
                        "%s request had no explicit GM/PINNED/IGNORE markers in system/user messages; "
                        "falling back to filtering message index/indices %s",
                        resolved_request_type,
                        target_indices,
                    )

            for target_index in target_indices:
                filtered = await prompt_filter.filter_prompt(
                    messages=messages_to_send,
                    request_type=resolved_request_type,
                    target_message_index=target_index,
                )

                stats = prompt_filter.get_filter_stats(filtered)
                stats_rollup["original_size"] += filtered.original_size
                stats_rollup["filtered_size"] += filtered.filtered_size
                stats_rollup["sections_included"] += filtered.sections_included
                stats_rollup["pinned_sections_included"] += filtered.pinned_sections_included
                target_roles.append(str(messages_to_send[target_index].get("role", "user")))
                filtered_target_indices.append(target_index)

                messages_to_send[target_index] = dict(messages_to_send[target_index])
                rewritten_content = _build_filtered_content_preserving_shape(
                    messages_to_send[target_index].get("content"),
                    filtered.system_prompt,
                )
                if rewritten_content is None:
                    logger.warning("Skipping in-place prompt rewrite for non-text structured content at message index %s", target_index)
                    continue
                messages_to_send[target_index]["content"] = rewritten_content

            reduced = stats_rollup["original_size"] - stats_rollup["filtered_size"]
            reduction_pct = ((reduced / stats_rollup["original_size"]) * 100) if stats_rollup["original_size"] else 0.0
            logger.info(
                "GM stats: %s",
                {
                    "request_type": resolved_request_type,
                    "target_roles": target_roles,
                    "target_indices": filtered_target_indices,
                    "original_size": stats_rollup["original_size"],
                    "filtered_size": stats_rollup["filtered_size"],
                    "reduction_pct": reduction_pct,
                    "sections_included": stats_rollup["sections_included"],
                    "pinned_sections_included": stats_rollup["pinned_sections_included"],
                },
            )
        else:
            messages_to_send = messages_dict
            logger.info("GM filtering disabled, passing through")
        
        # Add optional configured system prompts after prompt-container filtering, so they cannot
        # be selected as filter targets or confuse request-type detection.
        messages_to_send = _inject_configured_system_prompts(messages_to_send, resolved_request_type)
        messages_to_send = _strip_user_messages_for_special_dialogue_modes_if_enabled(
            messages_to_send,
            npc_to_npc_marker_present,
            group_marker_present,
        )

        # Build request for backend. Preserve provider/model-specific extra fields
        # such as tools, response_format, seed, transforms, route, etc.
        model = request.model
        if not model or model in {"gpt-4", "gpt-4-turbo"}:
            # If detection failed, get_model falls back to the configured default model.
            model = settings.get_model(resolved_request_type)

        backend_request = _build_backend_request(
            request=request,
            model=model,
            messages_to_send=messages_to_send,
            resolved_request_type=resolved_request_type,
        )

        endpoint_url = settings.get_chat_completions_url()
        headers = settings.get_headers()
        logger.debug(f"Forwarding request to {endpoint_url}")

        # Use httpx's JSON request-building path for the real outbound request, then log
        # the exact serialized bytes that will be sent on the wire.
        outbound_request = http_client.build_request(
            "POST",
            endpoint_url,
            json=backend_request,
            headers=headers,
        )
        request_content_bytes = outbound_request.content or b""
        wire_json = request_content_bytes.decode("utf-8", errors="replace")
        wire_chars = len(wire_json)
        wire_sha256 = hashlib.sha256(request_content_bytes).hexdigest()

        await append_llm_log(
            "REQUEST SENT TO LLM",
            request_id,
            {
                "endpoint_url": endpoint_url,
                "request_type": resolved_request_type,
                "request_type_hint": request.request_type,
                "use_gm": gm_filter_enabled,
                "transport_mode": "httpx_json_request",
                "wire_json_chars": wire_chars,
                "wire_json_sha256": wire_sha256,
                "message_summaries": _summarize_messages_for_log(messages_to_send),
                "payload": backend_request,
            },
        )

        response = await http_client.send(outbound_request)

        if response.status_code != 200:
            await append_llm_log(
                "RESPONSE FROM LLM ERROR",
                request_id,
                f"HTTP {response.status_code}\n{response.text}",
                raw=True,
            )
            logger.error(f"Backend error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Backend error: {response.text}"
            )

        await append_llm_log(
            "RESPONSE FROM LLM",
            request_id,
            response.text,
            raw=True,
        )
        result = response.json()
        provider_prompt_tokens = _extract_provider_prompt_tokens(result)
        local_prompt_tokens = _approx_prompt_tokens(messages_to_send)
        if (
            provider_prompt_tokens is not None
            and local_prompt_tokens >= 4000
            and provider_prompt_tokens < int(local_prompt_tokens * 0.35)
        ):
            logger.warning(
                "Prompt token mismatch detected for model=%s: provider reported %s tokens, local estimate is ~%s tokens.",
                model,
                provider_prompt_tokens,
                local_prompt_tokens,
            )
        repair_summaries = _repair_chat_completion_content(result)
        if repair_summaries:
            logger.info("JSON repair applied to request %s: %s", request_id, "; ".join(repair_summaries))
        if repair_summaries:
            await append_llm_log(
                "RESPONSE RETURNED TO GAME AFTER REPAIR",
                request_id,
                {
                    "local_prompt_tokens_estimate": local_prompt_tokens,
                    "provider_prompt_tokens_reported": provider_prompt_tokens,
                    "repair_summaries": repair_summaries,
                    "payload_after_repair": result,
                },
            )

        # Add timing info
        elapsed = time.time() - request_start
        logger.info(f"Request completed in {elapsed:.2f}s")

        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _require_character_memory_manager() -> CharacterMemoryManager:
    if character_memory_manager is None:
        raise HTTPException(status_code=503, detail="Character Memory manager is not initialized")
    return character_memory_manager


async def _reload_and_get_character_memory_manager(config_path: Optional[str] = None) -> CharacterMemoryManager:
    # Character Memory settings are edited from the GUI and should apply to the
    # very next button click, just like /reindex.  Reloading here is cheap and
    # does not rebuild or reindex the Static GM DB.
    async with runtime_lock:
        await _reload_runtime_locked(config_path)
        return _require_character_memory_manager()


@app.get("/character-memory/status")
async def character_memory_status():
    """Runtime status for Character Memory Control."""
    manager = _require_character_memory_manager()
    return manager.get_status()


@app.post("/character-memory/scan")
async def character_memory_scan(request: Request):
    """Scan campaign folder and report detected AIInfluence character JSONs."""
    manager = await _reload_and_get_character_memory_manager(await _request_config_path(request))
    return await asyncio.to_thread(manager.scan_campaign)


@app.post("/character-memory/backup")
async def character_memory_backup(request: Request):
    """Create BACKUP1/BACKUP2/etc. with all detected character JSONs."""
    manager = await _reload_and_get_character_memory_manager(await _request_config_path(request))
    try:
        return await asyncio.to_thread(manager.backup_campaign)
    except Exception as e:
        logger.error("Character Memory backup failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/character-memory/summarize")
async def character_memory_summarize(request: Request, req: CharacterMemoryActionRequest | None = None):
    """Summarize old ConversationHistory lines into MEMORY entries."""
    manager = await _reload_and_get_character_memory_manager(await _request_config_path(request))
    try:
        if req and req.create_backup:
            await asyncio.to_thread(manager.backup_campaign)
        return await asyncio.to_thread(manager.summarize_campaign, False)
    except Exception as e:
        logger.error("Character Memory summarize failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/character-memory/update-profile")
async def character_memory_update_profile(request: Request, req: CharacterMemoryActionRequest | None = None):
    """Conservatively update character personality/backstory from ConversationHistory."""
    manager = await _reload_and_get_character_memory_manager(await _request_config_path(request))
    try:
        if req and req.create_backup:
            await asyncio.to_thread(manager.backup_campaign)
        return await asyncio.to_thread(manager.update_profiles)
    except Exception as e:
        logger.error("Character Memory profile update failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex")
async def reindex(request: Request):
    """Reload current settings from disk, then manually force Static GM Index rebuild."""
    config_path = await _request_config_path(request)
    async with runtime_lock:
        try:
            logger.info("Starting forced reindex with freshly reloaded settings...")
            await _reload_runtime_locked(config_path)
            await retriever.reindex()
            return {
                "status": "success",
                "stats": retriever.get_stats(),
                "static_gm_index_summary_model": settings.static_gm_index_summary_model if settings else "",
                "static_gm_index_summary_prompt_chars": len(settings.static_gm_index_summary_instruction or "") if settings else 0,
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Reindex error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/sections")
async def debug_sections():
    """Get debug info about indexed sections."""
    sections_info = []
    
    for i, section in enumerate(retriever.sections[:20]):  # Limit to first 20
        sections_info.append({
            "index": i,
            "title": section.title,
            "source": section.source,
            "level": section.level,
            "policy": getattr(section, "policy", "gm"),
            "content_length": len(section.full_content),
            "summary": section.summary[:100] + "..." if len(section.summary) > 100 else section.summary,
            "entities": section.entities
        })
    
    return {
        "total_sections": len(retriever.sections),
        "pinned_world_sections": len(getattr(retriever, "pinned_sections", [])),
        "showing": len(sections_info),
        "sections": sections_info
    }


if __name__ == "__main__":
    import uvicorn
    
    # Load settings for startup
    settings = Settings.load()
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level="info"
    )
