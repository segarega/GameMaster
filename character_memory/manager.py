# Copyright (C) 2026 SEGAREGA
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Character Memory Control backend.

This module edits AIInfluence character JSON files in a campaign folder.  It is
intentionally separate from the Static GM Index: this system compresses
ConversationHistory at the source so AIInfluence itself later builds much smaller
prompts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


MEMORY_RE = re.compile(r"^\s*MEMORY\s*(\d+)\s*:\s*(.*)$", re.IGNORECASE | re.DOTALL)
DEFAULT_CHARACTER_SUMMARY_PROMPT = (
    "You are compressing a character's conversation history for long-term game memory. "
    "Summarize only the provided conversation lines. Preserve concrete facts, promises, threats, favors, secrets, "
    "relationships, conflicts, names, places, titles, allegiances, emotional shifts, debts, bargains, and information "
    "the character learned from or about the player. Preserve the character's attitude toward the player and any "
    "changes in trust, suspicion, respect, fear, anger, loyalty, or obligation. Do not invent events, motives, names, "
    "titles, or relationships. Do not contradict existing memory. Do not include trivial greetings, repeated phrasing, "
    "or generic banter unless it changed the relationship or revealed useful information. Write one concise paragraph "
    "in past tense. The paragraph must be usable as a MEMORY entry inside ConversationHistory. Do not use markdown, "
    "bullets, headings, or JSON."
)

DEFAULT_PROFILE_UPDATE_PROMPT = (
    "You are conservatively updating a game character profile using conversation history. "
    "You may update the character's personality or backstory only when the conversation reveals durable, meaningful "
    "information that should affect future roleplay. Examples include new relationships, loyalties, grudges, debts, "
    "promises, losses, family news, imprisonment, release, betrayal, alliance, fear, respect, or changed opinion of the "
    "player. Do not rewrite the character into a different person. Preserve their established temperament, social status, "
    "history, culture, speech style, values, and contradictions unless the conversation gives strong evidence of gradual "
    "change. Do not turn a cruel character kind, a cynical character trusting, a noble-born character lowborn, or a "
    "lifelong enemy into a friend without strong evidence. Prefer small additive edits over broad rewrites. Keep the "
    "existing structure and style where possible. If no meaningful durable change is needed, return changed=false. "
    "Return strict JSON only with this shape: {\"changed\": true or false, \"new_personality\": string or null, "
    "\"new_backstory\": string or null, \"reason\": string, \"confidence\": \"low\" | \"medium\" | \"high\"}."
)


@dataclass
class CharacterFile:
    path: Path
    name: str
    string_id: str
    history_count: int
    memory_count: int
    raw_count: int
    has_personality: bool
    has_backstory: bool


class CharacterMemoryError(RuntimeError):
    pass


class CharacterMemoryManager:
    """Manage AIInfluence campaign character memories."""

    def __init__(self, settings: Any):
        self.settings = settings
        self._operation_lock = threading.Lock()
        self._auto_task: Optional[asyncio.Task[Any]] = None
        self._auto_stop_event: Optional[asyncio.Event] = None
        self._last_result: Dict[str, Any] = {}
        self._last_scan: Dict[str, Any] = {}
        self._last_auto_run_utc: str = ""
        self._running_operation: str = ""

    # ------------------------------------------------------------------
    # Lifecycle / auto mode
    # ------------------------------------------------------------------
    async def start_auto(self) -> None:
        await self.stop_auto()
        if not bool(getattr(self.settings, "character_memory_auto_enabled", False)):
            return
        if not self._campaign_dir().exists():
            logger.info("Character Memory auto mode enabled, but campaign dir is not configured/found.")
            return
        self._auto_stop_event = asyncio.Event()
        self._auto_task = asyncio.create_task(self._auto_loop(), name="character-memory-auto")
        logger.info("Character Memory auto summarizer started")

    async def stop_auto(self) -> None:
        if self._auto_stop_event:
            self._auto_stop_event.set()
        if self._auto_task:
            self._auto_task.cancel()
            try:
                await self._auto_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning("Character Memory auto task stopped with error: %s", exc)
        self._auto_task = None
        self._auto_stop_event = None

    async def _auto_loop(self) -> None:
        stop = self._auto_stop_event
        if stop is None:
            return
        interval = max(5.0, float(getattr(self.settings, "character_memory_auto_scan_interval_seconds", 30.0) or 30.0))
        debounce = max(0.0, float(getattr(self.settings, "character_memory_auto_debounce_seconds", 8.0) or 8.0))
        while not stop.is_set():
            try:
                if debounce:
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=debounce)
                        break
                    except asyncio.TimeoutError:
                        pass
                result = await asyncio.to_thread(self.summarize_campaign, True)
                self._last_auto_run_utc = datetime.now(timezone.utc).isoformat()
                if result.get("updated_files"):
                    logger.info("Character Memory auto summarized %s files", result.get("updated_files"))
            except Exception as exc:
                logger.warning("Character Memory auto summarize failed: %s", exc, exc_info=True)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": bool(getattr(self.settings, "character_memory_enabled", True)),
            "campaign_dir": str(self._campaign_dir()),
            "campaign_dir_exists": self._campaign_dir().exists(),
            "auto_enabled": bool(getattr(self.settings, "character_memory_auto_enabled", False)),
            "auto_running": self._auto_task is not None and not self._auto_task.done(),
            "running_operation": self._running_operation,
            "last_auto_run_utc": self._last_auto_run_utc,
            "last_scan": self._last_scan,
            "last_result": self._last_result,
        }

    def scan_campaign(self) -> Dict[str, Any]:
        root = self._campaign_dir()
        files = self._scan_character_files(root)
        total_history = sum(item.history_count for item in files)
        total_raw = sum(item.raw_count for item in files)
        total_memory = sum(item.memory_count for item in files)
        trigger = int(getattr(self.settings, "character_memory_auto_trigger_raw_lines", 16) or 16)
        preserve = int(getattr(self.settings, "character_memory_preserve_last_lines", 10) or 10)
        result = {
            "status": "success",
            "campaign_dir": str(root),
            "campaign_dir_exists": root.exists(),
            "character_files": len(files),
            "with_conversation_history": sum(1 for item in files if item.history_count > 0),
            "total_history_lines": total_history,
            "total_memory_lines": total_memory,
            "total_raw_lines": total_raw,
            "over_auto_threshold": sum(1 for item in files if item.raw_count > trigger),
            "summarizable_now": sum(1 for item in files if item.raw_count > preserve),
            "files": [
                {
                    "file": str(item.path.relative_to(root)) if root.exists() else str(item.path),
                    "name": item.name,
                    "string_id": item.string_id,
                    "history_lines": item.history_count,
                    "memory_lines": item.memory_count,
                    "raw_lines": item.raw_count,
                    "has_personality": item.has_personality,
                    "has_backstory": item.has_backstory,
                }
                for item in files[:500]
            ],
        }
        self._last_scan = result
        return result

    def backup_campaign(self) -> Dict[str, Any]:
        root = self._campaign_dir()
        if not root.exists():
            raise CharacterMemoryError(f"Campaign folder does not exist: {root}")
        files = self._iter_campaign_json_files(root)
        next_index = 1
        while (root / f"BACKUP{next_index}").exists():
            next_index += 1
        backup_dir = root / f"BACKUP{next_index}"
        backup_dir.mkdir(parents=True, exist_ok=False)
        copied = 0
        errors: List[str] = []
        for path in files:
            try:
                rel = path.relative_to(root)
                target = backup_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied += 1
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "campaign_dir": str(root),
            "file_count": copied,
            "operation": "manual_backup",
            "errors": errors,
        }
        (backup_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        result = {"status": "success", "backup_dir": str(backup_dir), "file_count": copied, "errors": errors}
        self._last_result = result
        return result

    def summarize_campaign(self, auto: bool = False) -> Dict[str, Any]:
        if not self._operation_lock.acquire(blocking=False):
            return {"status": "busy", "message": f"Already running: {self._running_operation}"}
        self._running_operation = "auto_summarize" if auto else "summarize"
        started = time.time()
        try:
            root = self._campaign_dir()
            files = self._scan_character_files(root)
            preserve = max(0, int(getattr(self.settings, "character_memory_preserve_last_lines", 10) or 10))
            trigger = max(1, int(getattr(self.settings, "character_memory_auto_trigger_raw_lines", 16) or 16))
            result = {
                "status": "success",
                "mode": "auto" if auto else "manual",
                "campaign_dir": str(root),
                "files_seen": len(files),
                "updated_files": 0,
                "skipped_files": 0,
                "llm_calls": 0,
                "errors": [],
                "details": [],
                "elapsed_seconds": 0.0,
            }
            for item in files:
                try:
                    detail = self._summarize_file(item.path, preserve=preserve, trigger=trigger, auto=auto)
                    result["details"].append(detail)
                    result["llm_calls"] += int(detail.get("llm_calls", 0) or 0)
                    if detail.get("updated"):
                        result["updated_files"] += 1
                    else:
                        result["skipped_files"] += 1
                except Exception as exc:
                    result["errors"].append(f"{item.path.name}: {exc}")
            result["elapsed_seconds"] = round(time.time() - started, 3)
            self._last_result = result
            return result
        finally:
            self._running_operation = ""
            self._operation_lock.release()

    def update_profiles(self) -> Dict[str, Any]:
        if not self._operation_lock.acquire(blocking=False):
            return {"status": "busy", "message": f"Already running: {self._running_operation}"}
        self._running_operation = "update_profiles"
        started = time.time()
        try:
            root = self._campaign_dir()
            files = self._scan_character_files(root)
            result = {
                "status": "success",
                "campaign_dir": str(root),
                "files_seen": len(files),
                "updated_files": 0,
                "skipped_files": 0,
                "llm_calls": 0,
                "errors": [],
                "details": [],
                "elapsed_seconds": 0.0,
            }
            for item in files:
                try:
                    detail = self._update_profile_file(item.path)
                    result["details"].append(detail)
                    result["llm_calls"] += int(detail.get("llm_calls", 0) or 0)
                    if detail.get("updated"):
                        result["updated_files"] += 1
                    else:
                        result["skipped_files"] += 1
                except Exception as exc:
                    result["errors"].append(f"{item.path.name}: {exc}")
            result["elapsed_seconds"] = round(time.time() - started, 3)
            self._last_result = result
            return result
        finally:
            self._running_operation = ""
            self._operation_lock.release()

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------
    def _campaign_dir(self) -> Path:
        raw = str(getattr(self.settings, "character_memory_campaign_dir", "") or "").strip()
        if not raw:
            return Path("__character_memory_campaign_dir_not_configured__").resolve()
        return Path(raw).expanduser().resolve()

    def _iter_campaign_json_files(self, root: Path) -> List[Path]:
        if not root.exists() or not root.is_dir():
            return []
        files: List[Path] = []
        for path in sorted(root.rglob("*.json"), key=lambda p: str(p).lower()):
            try:
                rel_parts = [part.lower() for part in path.relative_to(root).parts[:-1]]
            except Exception:
                rel_parts = []
            if any(part.startswith("backup") for part in rel_parts):
                continue
            if path.name.lower() == "manifest.json":
                continue
            files.append(path)
        return files

    def _scan_character_files(self, root: Path) -> List[CharacterFile]:
        files: List[CharacterFile] = []
        for path in self._iter_campaign_json_files(root):
            try:
                data = self._read_json(path)
            except Exception:
                continue
            if not self._looks_like_character_json(data):
                continue
            history = data.get("ConversationHistory") if isinstance(data.get("ConversationHistory"), list) else []
            memory_lines, raw_lines = self._split_history(history)
            files.append(CharacterFile(
                path=path,
                name=str(data.get("Name") or path.stem),
                string_id=str(data.get("StringId") or ""),
                history_count=len(history),
                memory_count=len(memory_lines),
                raw_count=len(raw_lines),
                has_personality=bool(str(data.get("AIGeneratedPersonality") or "").strip()),
                has_backstory=bool(str(data.get("AIGeneratedBackstory") or "").strip()),
            ))
        return files

    def _looks_like_character_json(self, data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        if isinstance(data.get("ConversationHistory"), list):
            return True
        if "Name" in data and "StringId" in data and (
            "AIGeneratedPersonality" in data or "AIGeneratedBackstory" in data
        ):
            return True
        return False

    def _read_json(self, path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def _write_json_atomic(self, path: Path, data: Dict[str, Any], expected_stat: os.stat_result) -> None:
        current_stat = path.stat()
        if current_stat.st_mtime_ns != expected_stat.st_mtime_ns or current_stat.st_size != expected_stat.st_size:
            raise CharacterMemoryError("File changed while LLM operation was running; skipped to avoid overwriting new AIInfluence data")
        encoded = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(encoded)
            os.replace(temp_name, path)
        finally:
            try:
                if os.path.exists(temp_name):
                    os.remove(temp_name)
            except Exception:
                pass

    def _split_history(self, history: Iterable[Any]) -> Tuple[List[str], List[str]]:
        memory: List[str] = []
        raw: List[str] = []
        for item in history:
            text = str(item)
            if MEMORY_RE.match(text.strip()):
                memory.append(text)
            else:
                raw.append(text)
        return memory, raw

    def _next_memory_index(self, memory_lines: List[str]) -> int:
        max_index = 0
        for line in memory_lines:
            m = MEMORY_RE.match(line.strip())
            if m:
                try:
                    max_index = max(max_index, int(m.group(1)))
                except Exception:
                    pass
        return max_index + 1

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------
    def _summarize_file(self, path: Path, preserve: int, trigger: int, auto: bool) -> Dict[str, Any]:
        stat = path.stat()
        data = self._read_json(path)
        history = data.get("ConversationHistory")
        if not isinstance(history, list) or not history:
            return {"file": path.name, "updated": False, "reason": "no ConversationHistory", "llm_calls": 0}
        memory_lines, raw_lines = self._split_history(history)
        if auto and len(raw_lines) <= trigger:
            return {"file": path.name, "updated": False, "reason": "below auto trigger", "raw_lines": len(raw_lines), "llm_calls": 0}
        if len(raw_lines) <= preserve:
            return {"file": path.name, "updated": False, "reason": "nothing older than preserved lines", "raw_lines": len(raw_lines), "llm_calls": 0}
        old_raw = raw_lines[:-preserve] if preserve else list(raw_lines)
        preserved_raw = raw_lines[-preserve:] if preserve else []
        if not old_raw:
            return {"file": path.name, "updated": False, "reason": "no old raw lines", "raw_lines": len(raw_lines), "llm_calls": 0}

        summary = self._call_summary_llm(data, memory_lines, old_raw)
        llm_calls = 1
        summary = self._clean_memory_summary(summary)
        if not summary:
            return {"file": path.name, "updated": False, "reason": "empty summary", "llm_calls": llm_calls}

        new_memory_index = self._next_memory_index(memory_lines)
        new_memory_lines = list(memory_lines) + [f"MEMORY{new_memory_index}: {summary}"]
        max_memory_entries = max(1, int(getattr(self.settings, "character_memory_max_memory_entries", 5) or 5))
        compacted = False
        if len(new_memory_lines) > max_memory_entries:
            compacted_summary = self._call_memory_compaction_llm(data, new_memory_lines)
            llm_calls += 1
            compacted_summary = self._clean_memory_summary(compacted_summary)
            if compacted_summary:
                new_memory_lines = [f"MEMORY1: {compacted_summary}"]
                compacted = True

        data["ConversationHistory"] = new_memory_lines + preserved_raw
        self._write_json_atomic(path, data, stat)
        return {
            "file": path.name,
            "name": str(data.get("Name") or path.stem),
            "updated": True,
            "summarized_raw_lines": len(old_raw),
            "preserved_raw_lines": len(preserved_raw),
            "memory_lines": len(new_memory_lines),
            "compacted": compacted,
            "llm_calls": llm_calls,
        }

    def _call_summary_llm(self, data: Dict[str, Any], memory_lines: List[str], raw_lines: List[str]) -> str:
        name = str(data.get("Name") or "Unknown")
        string_id = str(data.get("StringId") or "")
        personality = str(data.get("AIGeneratedPersonality") or "")
        backstory = str(data.get("AIGeneratedBackstory") or "")
        prompt = str(getattr(self.settings, "character_memory_summary_prompt", "") or DEFAULT_CHARACTER_SUMMARY_PROMPT)
        user = (
            f"Character: {name}\n"
            f"StringId: {string_id}\n\n"
            f"Existing personality:\n{personality[:4000]}\n\n"
            f"Existing backstory:\n{backstory[:4000]}\n\n"
            f"Existing MEMORY entries:\n{self._join_lines(memory_lines, limit=6000) or '(none)'}\n\n"
            f"Conversation lines to summarize:\n{self._join_lines(raw_lines, limit=18000)}"
        )
        return self._chat_completion(prompt, user)

    def _call_memory_compaction_llm(self, data: Dict[str, Any], memory_lines: List[str]) -> str:
        prompt = (
            "Consolidate the provided MEMORY entries for one game character into one concise long-term memory paragraph. "
            "Preserve durable facts, relationship changes, promises, threats, secrets, debts, conflicts, and changed attitudes. "
            "Do not invent anything. Do not use markdown, bullets, headings, or JSON."
        )
        user = (
            f"Character: {data.get('Name') or 'Unknown'}\n"
            f"StringId: {data.get('StringId') or ''}\n\n"
            f"MEMORY entries to consolidate:\n{self._join_lines(memory_lines, limit=20000)}"
        )
        return self._chat_completion(prompt, user)

    def _clean_memory_summary(self, text: str) -> str:
        text = self._strip_code_fences(str(text or "")).strip()
        text = re.sub(r"^\s*MEMORY\s*\d+\s*:\s*", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    # Profile update
    # ------------------------------------------------------------------
    def _update_profile_file(self, path: Path) -> Dict[str, Any]:
        stat = path.stat()
        data = self._read_json(path)
        history = data.get("ConversationHistory")
        if not isinstance(history, list) or not history:
            return {"file": path.name, "updated": False, "reason": "no ConversationHistory", "llm_calls": 0}
        joined_history = self._join_lines([str(x) for x in history], limit=22000)
        if not joined_history.strip():
            return {"file": path.name, "updated": False, "reason": "empty ConversationHistory", "llm_calls": 0}
        response = self._call_profile_update_llm(data, joined_history)
        parsed = self._parse_json_object(response)
        changed = bool(parsed.get("changed", False)) if isinstance(parsed, dict) else False
        if not changed:
            return {
                "file": path.name,
                "name": str(data.get("Name") or path.stem),
                "updated": False,
                "reason": str(parsed.get("reason", "no durable change") if isinstance(parsed, dict) else "no durable change"),
                "confidence": str(parsed.get("confidence", "") if isinstance(parsed, dict) else ""),
                "llm_calls": 1,
            }
        new_personality = parsed.get("new_personality") if isinstance(parsed, dict) else None
        new_backstory = parsed.get("new_backstory") if isinstance(parsed, dict) else None
        changed_fields: List[str] = []
        if isinstance(new_personality, str) and new_personality.strip():
            old = str(data.get("AIGeneratedPersonality") or "")
            if new_personality.strip() != old.strip():
                data["AIGeneratedPersonality"] = new_personality.strip()
                changed_fields.append("AIGeneratedPersonality")
        if isinstance(new_backstory, str) and new_backstory.strip():
            old = str(data.get("AIGeneratedBackstory") or "")
            if new_backstory.strip() != old.strip():
                data["AIGeneratedBackstory"] = new_backstory.strip()
                changed_fields.append("AIGeneratedBackstory")
        if not changed_fields:
            return {"file": path.name, "updated": False, "reason": "LLM marked changed but returned no changed fields", "llm_calls": 1}
        self._write_json_atomic(path, data, stat)
        return {
            "file": path.name,
            "name": str(data.get("Name") or path.stem),
            "updated": True,
            "changed_fields": changed_fields,
            "reason": str(parsed.get("reason", "") if isinstance(parsed, dict) else ""),
            "confidence": str(parsed.get("confidence", "") if isinstance(parsed, dict) else ""),
            "llm_calls": 1,
        }

    def _call_profile_update_llm(self, data: Dict[str, Any], joined_history: str) -> str:
        prompt = str(getattr(self.settings, "character_memory_profile_update_prompt", "") or DEFAULT_PROFILE_UPDATE_PROMPT)
        user = (
            f"Name: {data.get('Name') or 'Unknown'}\n"
            f"StringId: {data.get('StringId') or ''}\n\n"
            f"Current AIGeneratedPersonality:\n{str(data.get('AIGeneratedPersonality') or '')[:8000]}\n\n"
            f"Current AIGeneratedBackstory:\n{str(data.get('AIGeneratedBackstory') or '')[:8000]}\n\n"
            f"ConversationHistory / MEMORY entries:\n{joined_history}"
        )
        return self._chat_completion(prompt, user)

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------
    def _effective_api_url(self) -> str:
        raw = str(getattr(self.settings, "character_memory_api_url", "") or "").strip()
        if not raw:
            raw = str(getattr(self.settings, "selector_api_url", "") or getattr(self.settings, "api_url", "") or "").strip()
        if not raw:
            raise CharacterMemoryError("Character Memory API URL is empty")
        base = raw.rstrip("/")
        if base.lower().endswith("/chat/completions"):
            return base
        if base.lower().endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/chat/completions"

    def _effective_api_key(self) -> str:
        return str(
            getattr(self.settings, "character_memory_api_key", "")
            or getattr(self.settings, "selector_api_key", "")
            or getattr(self.settings, "api_key", "")
            or ""
        )

    def _effective_model(self) -> str:
        model = str(getattr(self.settings, "character_memory_model", "") or "").strip()
        if model:
            return model
        model = str(getattr(self.settings, "selector_model", "") or "").strip()
        if model:
            return model
        models = getattr(self.settings, "models", {}) or {}
        if isinstance(models, dict):
            return next((str(v) for v in models.values() if str(v).strip()), "")
        return ""

    def _chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        model = self._effective_model()
        if not model:
            raise CharacterMemoryError("Character Memory model is empty")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        api_key = self._effective_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        site_url = str(getattr(self.settings, "site_url", "") or "").strip()
        app_title = str(getattr(self.settings, "app_title", "") or "GameMaster").strip() or "GameMaster"
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_title:
            headers["X-Title"] = app_title
            headers["X-OpenRouter-Title"] = app_title

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(getattr(self.settings, "character_memory_temperature", 0.1) or 0.1),
            "max_tokens": int(getattr(self.settings, "character_memory_max_tokens", 700) or 700),
            "stream": False,
        }
        timeout = float(getattr(self.settings, "character_memory_timeout_seconds", 180.0) or 180.0)
        with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
            response = client.post(self._effective_api_url(), headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        choices = data.get("choices") if isinstance(data, dict) else None
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    return content
            text = choices[0].get("text") if isinstance(choices[0], dict) else None
            if isinstance(text, str):
                return text
        raise CharacterMemoryError("Character Memory LLM response did not contain text")

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def _join_lines(self, lines: Iterable[str], limit: int) -> str:
        out: List[str] = []
        used = 0
        for line in lines:
            text = str(line)
            if used + len(text) + 1 > limit:
                remaining = max(0, limit - used - 4)
                if remaining > 0:
                    out.append(text[:remaining].rstrip() + "...")
                break
            out.append(text)
            used += len(text) + 1
        return "\n".join(out)

    def _strip_code_fences(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^\s*```(?:json|JSON|[a-zA-Z0-9_-]+)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        return cleaned.strip()

    def _parse_json_object(self, text: str) -> Dict[str, Any]:
        cleaned = self._strip_code_fences(str(text or ""))
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}
