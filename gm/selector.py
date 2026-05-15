# Copyright (C) 2026 SEGAREGA
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Proxy-side selector model client for [GM] section decisions.

This client sends a compact context packet plus candidate indexed entries to an
OpenAI-compatible model. The model returns only section IDs to keep; the proxy
reconstructs the exact original text deterministically.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from config.settings import Settings

logger = logging.getLogger(__name__)

FLAT_KEEP_IDS_BLOCK = "__flat_keep_ids__"


class GMSelectorClient:
    """OpenAI-compatible client for proxy-side [GM] section selection."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = bool(
            getattr(settings, "selector_enabled", False)
            and getattr(settings, "selector_api_url", "")
            and getattr(settings, "selector_model", "")
        )
        self.api_url = str(getattr(settings, "selector_api_url", "") or "").strip()
        self.api_key = str(getattr(settings, "selector_api_key", "") or "")
        self.model = str(getattr(settings, "selector_model", "") or "").strip()
        self.temperature = float(getattr(settings, "selector_temperature", 0.0))
        self.max_tokens = int(getattr(settings, "selector_max_tokens", 1200))
        self.timeout_seconds = float(getattr(settings, "selector_timeout_seconds", 120.0))
        self.system_prompt = str(getattr(settings, "selector_instruction", "") or "").strip()
        self.log_enabled = bool(getattr(settings, "selector_log_enabled", False))
        self.log_path = str(getattr(settings, "selector_log_path", "logs/selector-log.txt") or "logs/selector-log.txt")
        self.log_pretty_json = bool(getattr(settings, "selector_log_pretty_json", True))
        self.client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        """Create the selector HTTP client."""
        if not self.enabled:
            logger.info("Selector model disabled")
            return
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_seconds))
        logger.info("Selector model enabled: %s", self.model)

    async def aclose(self) -> None:
        """Close HTTP resources."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    def get_chat_completions_url(self) -> str:
        """Return the final OpenAI-compatible chat/completions URL."""
        url = self.api_url.rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        return f"{url}/chat/completions"

    def get_headers(self) -> Dict[str, str]:
        """Selector request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_stats(self) -> Dict[str, Any]:
        """Return selector status for diagnostics."""
        return {
            "enabled": self.enabled,
            "api_url": self.get_chat_completions_url() if self.enabled else "",
            "model": self.model if self.enabled else "",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "client_ready": self.client is not None,
            "log_enabled": self.log_enabled,
            "log_path": self.log_path,
        }

    async def select_relevant_section_ids(
        self,
        *,
        request_type: str,
        dialogue_query: str,
        context_segments: Sequence[Tuple[str, str]],
        block_title: str,
        sections: Sequence[Dict[str, str]],
    ) -> List[str]:
        """Backward-compatible wrapper for a single [GM] block selection."""
        block_decisions = await self.select_relevant_blocks(
            request_type=request_type,
            dialogue_query=dialogue_query,
            context_segments=context_segments,
            blocks=[
                {
                    "id": "block_1",
                    "title": block_title,
                    "sections": list(sections),
                }
            ],
        )
        return block_decisions.get("block_1", [])

    async def select_relevant_blocks(
        self,
        *,
        request_type: str,
        dialogue_query: str,
        context_segments: Sequence[Tuple[str, str]],
        blocks: Sequence[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """Return entry IDs to keep for one or more candidate blocks."""
        if not self.enabled or not blocks:
            return {}
        if self.client is None:
            raise RuntimeError("Selector model client not initialized")

        user_prompt = self._build_user_prompt(
            request_type=request_type,
            dialogue_query=dialogue_query,
            context_segments=context_segments,
            blocks=blocks,
        )
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        await self._append_selector_log(
            "SELECTOR REQUEST",
            {
                "endpoint_url": self.get_chat_completions_url(),
                "request_type": request_type,
                "block_count": len(blocks),
                "blocks": [
                    {
                        "id": str(block.get("id", "") or ""),
                        "title": str(block.get("title", "") or ""),
                        "candidate_count": len(block.get("sections") or []),
                    }
                    for block in blocks
                ],
                "context_segments": [
                    {"label": label, "chars": len(text), "text": text}
                    for label, text in context_segments
                ],
                "payload": request_body,
            },
        )

        response = await self.client.post(
            self.get_chat_completions_url(),
            json=request_body,
            headers=self.get_headers(),
        )
        if response.status_code != 200:
            await self._append_selector_log(
                "SELECTOR RESPONSE ERROR",
                f"HTTP {response.status_code}\n{response.text}",
                raw=True,
            )
            response.raise_for_status()
        await self._append_selector_log("SELECTOR RESPONSE", response.text, raw=True)
        payload = response.json()

        content = self._extract_response_text(payload)
        if not content:
            raise ValueError("Selector model returned no content")

        parsed_ok, keep_by_block = self._parse_keep_ids_by_block(content)
        if not parsed_ok:
            raise ValueError(f"Selector model returned invalid JSON: {content[:300]}")

        valid_ids_by_block: Dict[str, set[str]] = {}
        for block in blocks:
            block_id = str(block.get("id", "") or "").strip()
            valid_ids_by_block[block_id] = {
                str(section.get("id", "") or "").strip()
                for section in (block.get("sections") or [])
                if isinstance(section, dict)
            }

        flat_keep_ids = keep_by_block.pop(FLAT_KEEP_IDS_BLOCK, [])
        if flat_keep_ids:
            remaining = list(flat_keep_ids)
            for block_id, valid_ids in valid_ids_by_block.items():
                matched = [section_id for section_id in remaining if section_id in valid_ids]
                if matched:
                    keep_by_block[block_id] = list(dict.fromkeys(keep_by_block.get(block_id, []) + matched))
                    remaining = [section_id for section_id in remaining if section_id not in valid_ids]

        filtered_keep_by_block: Dict[str, List[str]] = {}
        for block_id, valid_ids in valid_ids_by_block.items():
            selected_ids = keep_by_block.get(block_id, [])
            filtered_ids = [section_id for section_id in selected_ids if section_id in valid_ids]
            if len(filtered_ids) != len(selected_ids):
                logger.warning("Selector returned unknown IDs for block %s; ignoring invalid entries", block_id)
            filtered_keep_by_block[block_id] = filtered_ids

        await self._append_selector_log(
            "SELECTOR DECISION",
            {
                "blocks": [
                    {
                        "id": str(block.get("id", "") or ""),
                        "title": str(block.get("title", "") or ""),
                        "keep_ids": filtered_keep_by_block.get(str(block.get("id", "") or ""), []),
                    }
                    for block in blocks
                ],
            },
        )
        return filtered_keep_by_block

    def _build_user_prompt(
        self,
        *,
        request_type: str,
        dialogue_query: str,
        context_segments: Sequence[Tuple[str, str]],
        blocks: Sequence[Dict[str, Any]],
    ) -> str:
        """Format the selector payload as a compact, deterministic text packet."""
        parts: List[str] = []

        if context_segments:
            parts.append("### CONTEXT EXTRACTS ###")
            for label, text in context_segments:
                if not text.strip():
                    continue
                parts.append(f"=== {label} ===")
                parts.append(text.strip())
        elif dialogue_query.strip():
            parts.append("### CONTEXT EXTRACTS ###")
            parts.append("=== Current User Message ===")
            parts.append(dialogue_query.strip())

        parts.append("### CANDIDATE INDEXED ENTRIES ###")
        for block in blocks:
            block_id = str(block.get("id", "") or "").strip() or "block"
            block_title = str(block.get("title", "") or "").strip() or "Candidate Block"
            parts.append(f'<BLOCK id="{block_id}" title="{block_title}">')
            for section in (block.get("sections") or []):
                section_id = str(section.get("id", "") or "").strip()
                section_title = str(section.get("title", "") or "").strip()
                section_parent = str(section.get("parent", "") or "").strip()
                summary = str(section.get("summary", "") or "").strip()
                content = str(section.get("content", "") or "").strip()
                attrs = [f'id="{section_id}"']
                if section_title:
                    attrs.append(f'title="{section_title}"')
                # Parent is already represented by the surrounding BLOCK title in
                # normal Static GM summary mode.  Include it only for unusual mixed
                # blocks where it differs, and do not send source file names at all.
                if section_parent and section_parent != block_title:
                    attrs.append(f'parent="{section_parent}"')
                parts.append(f'<SECTION {" ".join(attrs)}>')
                if summary:
                    parts.append("Summary:")
                    parts.append(summary)
                elif content:
                    parts.append(content)
                parts.append("</SECTION>")
            parts.append("</BLOCK>")

        parts.extend(
            [
                "### REQUIRED OUTPUT ###",
                'Return JSON only: {"blocks":[{"block_id":"<one of the provided BLOCK ids>","keep_ids":["<selected SECTION ids>"]}]}',
            ]
        )
        return "\n\n".join(part for part in parts if part)

    def _json_for_log(self, value: Any) -> str:
        """Serialize selector log content."""
        try:
            if self.log_pretty_json:
                return json.dumps(value, ensure_ascii=False, indent=2)
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(value)

    def _raw_content_for_log(self, content: Any) -> str:
        """Render raw selector content, prettifying JSON strings when requested."""
        text = content if isinstance(content, str) else str(content)
        if self.log_pretty_json:
            try:
                return self._json_for_log(json.loads(text))
            except Exception:
                pass
        return text

    def _log_block_text(self, label: str, content: str) -> str:
        timestamp = datetime.now(timezone.utc).isoformat()
        return (
            f"\n\n===== {label} | utc={timestamp} =====\n"
            f"{content}\n"
            f"===== END {label} =====\n"
        )

    def _append_text_file(self, path: str, text: str) -> None:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(text)

    async def _append_selector_log(self, label: str, content: Any, *, raw: bool = False) -> None:
        """Append selector request/response logs to a dedicated file."""
        if not self.log_enabled:
            return
        try:
            text = self._raw_content_for_log(content) if raw else self._json_for_log(content)
            await asyncio.to_thread(
                self._append_text_file,
                self.log_path,
                self._log_block_text(label, text),
            )
        except Exception as exc:
            logger.warning("Failed to write selector log: %s", exc)

    async def log_diagnostic(self, label: str, content: Any, *, raw: bool = False) -> None:
        """Public helper for writing extra selector-related diagnostics."""
        await self._append_selector_log(label, content, raw=raw)

    def _extract_response_text(self, payload: Any) -> str:
        """Extract assistant content from a chat completion response."""
        if not isinstance(payload, dict):
            return ""
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        message = first.get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
            return "\n".join(text_parts).strip()
        return ""

    def _parse_keep_ids_by_block(self, content: str) -> tuple[bool, Dict[str, List[str]]]:
        """Extract keep_ids grouped by block from the selector model response."""
        text = content.strip()
        if not text:
            return False, {}
        json_obj = self._extract_first_json_object(text)
        if not json_obj:
            logger.warning("Selector response was not JSON: %s", text[:300])
            return False, {}
        try:
            parsed = json.loads(json_obj)
        except Exception as exc:
            logger.warning("Failed to parse selector JSON response: %s", exc)
            return False, {}

        keep_by_block: Dict[str, List[str]] = {}

        if isinstance(parsed, dict):
            raw_blocks = parsed.get("blocks")
            if isinstance(raw_blocks, list):
                for block in raw_blocks:
                    if not isinstance(block, dict):
                        continue
                    block_id = str(block.get("block_id") or block.get("id") or "").strip()
                    if not block_id:
                        continue
                    keep_ids = self._coerce_keep_id_list(
                        block.get("keep_ids") or block.get("selected_ids") or block.get("ids") or block.get("keep")
                    )
                    keep_by_block[block_id] = keep_ids
                return True, keep_by_block

            raw_map = parsed.get("keep_by_block") or parsed.get("selected_by_block")
            if isinstance(raw_map, dict):
                for block_id, values in raw_map.items():
                    norm_block_id = str(block_id or "").strip()
                    if not norm_block_id:
                        continue
                    keep_by_block[norm_block_id] = self._coerce_keep_id_list(values)
                return True, keep_by_block

            for key in ("keep_ids", "selected_ids", "ids", "keep"):
                value = parsed.get(key)
                if isinstance(value, list):
                    keep_by_block[FLAT_KEEP_IDS_BLOCK] = self._coerce_keep_id_list(value)
                    return True, keep_by_block

        return False, {}

    def _coerce_keep_id_list(self, candidates: Any) -> List[str]:
        """Normalize a raw selector keep-id list."""
        if not isinstance(candidates, list):
            return []
        cleaned: List[str] = []
        seen = set()
        for value in candidates:
            if not isinstance(value, str):
                continue
            norm = value.strip()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            cleaned.append(norm)
        return cleaned

    def _extract_first_json_object(self, text: str) -> str:
        """Return the first balanced JSON object substring from text."""
        start = text.find("{")
        if start < 0:
            return ""
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
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
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return ""
