# Copyright (C) 2026 SEGAREGA
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Static [GM] index for AIInfluence-authored text files.

This module builds a persistent SQLite cache from the editable AIInfluence
files.  It indexes only [GM] child elements, normally the `## Child`
entries under a top-level `=== [GM...] Parent ===` header.  The proxy then
uses this cache to send compact summaries to the selector model instead of
sending every full [GM] body.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

from config.settings import Settings

logger = logging.getLogger(__name__)


DEFAULT_SUMMARY_INSTRUCTION = (
    'You are summarizing one indexed text entry so another AI can later decide whether this entry is relevant to a request. '
    'Return only one concise paragraph, ideally 45-80 words. '
    "Preserve the entry's core meaning, important details, constraints, context, terms that may be useful for matching, "
    'and any details that would change when the full entry should or should not be included. '
    'Do not invent, generalize beyond the text, or add outside knowledge. Do not mention that this is a summary. '
    'Do not use markdown, bullets, headings, or labels unless they are necessary to preserve meaning. '
    'Compress aggressively, but keep enough concrete detail that the later AI can reliably decide whether the full entry should be included.'
)


@dataclass
class IndexedGMElement:
    id: str
    source_file: str
    source_path: str
    parent_title: str
    child_title: str
    scopes: List[str]
    full_content: str
    summary: str
    deterministic_summary: str
    content_hash: str
    keywords: List[str]
    order_index: int
    summary_source: str = "deterministic"
    score: float = 0.0
    reason: str = ""


class StaticGMIndex:
    """Persistent local cache for editable AIInfluence [GM] elements."""

    POLICY_HEADER_RE = re.compile(
        r'^\s*(?:(?P<eq>={3,})\s*\[(?P<policy_eq>PINNED|PIN|GM|IGNORE)(?P<scopes_eq>(?::[A-Z0-9_ -]+)*)\]\s*(?P<title_eq>.*?)\s*(?P=eq)|\[(?P<policy_bare>PINNED|PIN|GM|IGNORE)(?P<scopes_bare>(?::[A-Z0-9_ -]+)*)\]\s*(?P<title_bare>.*?))\s*$',
        re.IGNORECASE | re.MULTILINE,
    )
    END_POLICY_RE = re.compile(
        r'^\s*(?:={3,}\s*)?\[END\s+(?:PINNED|PIN|GM|IGNORE)(?::[A-Z0-9_ -]+)*\](?:\s*={3,})?\s*\.?\s*$',
        re.IGNORECASE | re.MULTILINE,
    )
    INLINE_POLICY_END_RE = re.compile(
        r'\s*\[END\s+(?:PINNED|PIN|GM|IGNORE)(?::[A-Z0-9_ -]+)*\]\s*\.?\s*$',
        re.IGNORECASE,
    )
    CHILD_HEADER_RE = re.compile(r'^\s*(#{2,6})\s+(.+?)\s*$', re.MULTILINE)
    TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_'-]{2,}")

    DEFAULT_FILES = [
        "world.txt",
        "actionrules.txt",
        "battlecombatrules.txt",
        "eventsanalyzerrules.txt",
        "eventsgeneratorrules.txt",
        "kingdomstatementrules.txt",
    ]

    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = bool(getattr(settings, "static_gm_index_enabled", False))
        self.ai_influence_folder = str(getattr(settings, "static_gm_index_ai_influence_folder", "") or "").strip()
        self.files = list(getattr(settings, "static_gm_index_files", None) or self.DEFAULT_FILES)
        self.db_path = self._resolve_path(str(getattr(settings, "static_gm_index_db_path", "cache/static_gm_index.sqlite3") or "cache/static_gm_index.sqlite3"))
        # Reindexing is intentionally manual-only. Settings reload/startup must never
        # create LLM summary calls or rebuild the DB unexpectedly.
        self.auto_reindex = False
        self.summary_enabled = bool(getattr(settings, "static_gm_index_summary_enabled", True))
        self.summary_max_chars = int(getattr(settings, "static_gm_index_summary_max_chars", 6000))
        self.summary_model = str(getattr(settings, "static_gm_index_summary_model", "") or "").strip()
        self.summary_api_url = str(getattr(settings, "static_gm_index_summary_api_url", "") or "").strip()
        self.summary_api_key = str(getattr(settings, "static_gm_index_summary_api_key", "") or "")
        self.summary_temperature = float(getattr(settings, "static_gm_index_summary_temperature", 0.1))
        self.summary_max_tokens = int(getattr(settings, "static_gm_index_summary_max_tokens", 220))
        self.summary_timeout_seconds = float(getattr(settings, "static_gm_index_summary_timeout_seconds", 120.0))
        self.summary_instruction = str(getattr(settings, "static_gm_index_summary_instruction", "") or "").strip()
        self._memory: Dict[str, IndexedGMElement] = {}
        self._hash_to_ids: Dict[str, List[str]] = {}
        self._title_to_ids: Dict[Tuple[str, str], List[str]] = {}
        self._stats: Dict[str, Any] = {
            "enabled": self.enabled,
            "loaded": False,
            "db_path": str(self.db_path),
            "ai_influence_folder": self.ai_influence_folder,
            "files_configured": self.files,
            "elements": 0,
            "last_reindex": None,
        }

    async def initialize(self) -> None:
        if not self.enabled:
            logger.info("Static GM index disabled")
            return
        await asyncio.to_thread(self._ensure_schema)
        await self.load()
        if self._should_reindex():
            logger.info(
                "Static GM index needs manual reindex (DB missing/empty or source files changed). "
                "Use the GUI Save & Reindex DB button or POST /reindex. No automatic reindex was started."
            )
        logger.info("Static GM index ready: %s", self.get_stats())

    async def load(self) -> None:
        if not self.enabled:
            return
        await asyncio.to_thread(self._load_sync)

    async def reindex(self) -> Dict[str, Any]:
        if not self.enabled:
            self._stats.update({"enabled": False, "status": "disabled"})
            return self.get_stats()
        result = await self._reindex_async()
        await self.load()
        return result

    async def aclose(self) -> None:
        self._memory.clear()
        self._hash_to_ids.clear()
        self._title_to_ids.clear()

    def get_stats(self) -> Dict[str, Any]:
        stats = dict(self._stats)
        stats["elements"] = len(self._memory)
        stats["db_path"] = str(self.db_path)
        stats["ai_influence_folder"] = self.ai_influence_folder
        stats["files_configured"] = list(self.files)
        stats["summary_model"] = self._effective_summary_model()
        stats["summary_enabled"] = self.summary_enabled
        return stats

    def rank_prompt_sections(
        self,
        *,
        sections: Sequence[Any],
        query: str,
        entities: Dict[str, List[str]],
        request_type: str,
    ) -> List[Any]:
        """Prepare every eligible incoming [GM] child section for selector summary mode.

        This function intentionally does *not* shortlist by lexical score. The selector is
        the decision maker. The static index only attaches stable IDs, parent metadata,
        source file metadata, and cached summaries so the selector can choose from all
        available [GM] candidates without seeing full bodies.

        """
        if not sections:
            return []

        prepared: List[Any] = []
        parent_title = ""
        incoming_count = 0

        for section in sections:
            incoming_count += 1
            level = int(getattr(section, "level", 0) or 0)
            if level == 1:
                parent_title = str(getattr(section, "title", "") or "").strip()
                # Empty [GM] parents are containers only. Their children remain candidates.
                if self._is_empty_section(section):
                    continue

            if self._is_empty_section(section):
                continue

            indexed = self.find_for_prompt_section(section, parent_title=parent_title)
            summary = indexed.summary if indexed and indexed.summary else str(getattr(section, "summary", "") or "")
            if not summary:
                summary = self._deterministic_summary(
                    title=str(getattr(section, "title", "") or ""),
                    parent_title=parent_title,
                    full_content=str(getattr(section, "full_content", "") or ""),
                )

            setattr(section, "selector_summary", summary)
            setattr(section, "selector_parent_title", parent_title)
            setattr(section, "selector_match_score", 1.0)
            setattr(section, "selector_match_reason", "all_gm_candidates")

            if indexed:
                setattr(section, "static_index_id", indexed.id)
                setattr(section, "static_index_source", indexed.source_file)
                setattr(section, "static_index_summary_source", indexed.summary_source)

            prepared.append(section)

        logger.info(
            "Static GM index prepared all %d/%d eligible prompt [GM] sections for selector",
            len(prepared), incoming_count,
        )
        return prepared

    def find_for_prompt_section(self, section: Any, *, parent_title: str = "") -> Optional[IndexedGMElement]:
        full_content = str(getattr(section, "full_content", "") or "")
        content_hash = self._hash_text(self._canonical_content(full_content))
        by_hash = self._hash_to_ids.get(content_hash) or []
        if by_hash:
            return self._memory.get(by_hash[0])

        child_key = (
            self._slug(parent_title),
            self._slug(str(getattr(section, "title", "") or "")),
        )
        ids = self._title_to_ids.get(child_key) or []
        if ids:
            return self._memory.get(ids[0])
        return None

    def elements_for_request(self, request_type: str) -> List[IndexedGMElement]:
        """Return all indexed [GM] child elements whose scopes apply to this request.

        The static DB is the authoritative selector-candidate source.  Incoming
        AIInfluence prompt blocks are still used for live data and pinned text,
        but selector candidates should include every indexed GM child from every
        configured file, not only the entries that AIInfluence happened to inject
        into the current prompt.
        """
        aliases = self._request_type_aliases(request_type)
        out: List[IndexedGMElement] = []
        for elem in sorted(self._memory.values(), key=lambda e: e.order_index):
            if self._scopes_apply(elem.scopes, aliases):
                out.append(elem)
        return out

    def _request_type_aliases(self, request_type: Optional[str]) -> set:
        raw = (request_type or "").lower().strip()
        norm = re.sub(r'[^a-z0-9]+', '_', raw).strip('_')
        aliases = {norm} if norm else set()
        if norm in {"chat", "dialog", "dialogue"}:
            aliases.update({"dialog", "dialogue"})
        if "event" in norm:
            aliases.update({"event", "events", "event_generation", "eventgenerator", "event_generator"})
        if "diplom" in norm or "statement" in norm:
            aliases.update({"diplomacy", "diplomatic", "statement", "statements", "kingdom_statement"})
        if "battle" in norm or "combat" in norm:
            aliases.update({"battle", "combat", "battle_combat", "battlecombatrules"})
        return {a for a in aliases if a}

    def _scopes_apply(self, scopes: Sequence[str], aliases: set) -> bool:
        normalized = {re.sub(r'[^a-z0-9]+', '_', str(scope or '').lower()).strip('_') for scope in (scopes or [])}
        normalized = {scope for scope in normalized if scope}
        if not normalized:
            return True
        if normalized.intersection({"all", "any", "global", "default"}):
            return True
        return not aliases.isdisjoint(normalized)

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS gm_elements (
                    id TEXT PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    parent_title TEXT NOT NULL,
                    child_title TEXT NOT NULL,
                    scopes_json TEXT NOT NULL,
                    full_content TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    deterministic_summary TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    keywords_json TEXT NOT NULL,
                    order_index INTEGER NOT NULL,
                    summary_source TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_gm_hash ON gm_elements(content_hash)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_gm_title ON gm_elements(parent_title, child_title)")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS gm_index_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            db.commit()

    def _load_sync(self) -> None:
        self._ensure_schema()
        memory: Dict[str, IndexedGMElement] = {}
        hash_to_ids: Dict[str, List[str]] = {}
        title_to_ids: Dict[Tuple[str, str], List[str]] = {}
        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM gm_elements ORDER BY order_index ASC").fetchall()
            meta_rows = db.execute("SELECT key, value FROM gm_index_meta").fetchall()
        for row in rows:
            try:
                scopes = json.loads(row["scopes_json"] or "[]")
            except Exception:
                scopes = []
            try:
                keywords = json.loads(row["keywords_json"] or "[]")
            except Exception:
                keywords = []
            elem = IndexedGMElement(
                id=row["id"],
                source_file=row["source_file"],
                source_path=row["source_path"],
                parent_title=row["parent_title"],
                child_title=row["child_title"],
                scopes=[str(s) for s in scopes],
                full_content=row["full_content"],
                summary=row["summary"],
                deterministic_summary=row["deterministic_summary"],
                content_hash=row["content_hash"],
                keywords=[str(k) for k in keywords],
                order_index=int(row["order_index"]),
                summary_source=row["summary_source"],
            )
            memory[elem.id] = elem
            hash_to_ids.setdefault(elem.content_hash, []).append(elem.id)
            title_to_ids.setdefault((self._slug(elem.parent_title), self._slug(elem.child_title)), []).append(elem.id)
        self._memory = memory
        self._hash_to_ids = hash_to_ids
        self._title_to_ids = title_to_ids
        meta = {row["key"]: row["value"] for row in meta_rows}
        self._stats.update({
            "enabled": self.enabled,
            "loaded": True,
            "elements": len(memory),
            "last_reindex": meta.get("last_reindex"),
            "last_reindex_result": json.loads(meta.get("last_reindex_result", "{}") or "{}"),
        })

    async def _reindex_async(self) -> Dict[str, Any]:
        self._ensure_schema()
        old_by_hash = await asyncio.to_thread(self._old_summaries_by_hash)
        files = self._configured_paths()
        started = time.time()
        parsed: List[IndexedGMElement] = []
        stats = {
            "status": "success",
            "files_seen": 0,
            "files_missing": [],
            "gm_parent_blocks": 0,
            "gm_child_elements": 0,
            "pinned_blocks": 0,
            "ignored_blocks": 0,
            "summaries_llm_reused": 0,
            "summaries_llm_created": 0,
            "summaries_deterministic": 0,
            "errors": [],
        }

        order = 0
        for path in files:
            if not path.exists():
                stats["files_missing"].append(str(path))
                continue
            stats["files_seen"] += 1
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                blocks = self._parse_policy_blocks(text, path)
                for block in blocks:
                    policy = block["policy"]
                    if policy == "pinned":
                        stats["pinned_blocks"] += 1
                        continue
                    if policy == "ignore":
                        stats["ignored_blocks"] += 1
                        continue
                    stats["gm_parent_blocks"] += 1
                    block_body = block["body"]
                    if not str(block_body or "").strip() and block.get("inline_closed"):
                        # Compact one-line [GM] entries, e.g. [GM] Rule text [END GM],
                        # use the post-marker title text itself as the selectable content.
                        block_body = block["title"]
                    for child_title, child_content in self._split_gm_children(block_body, block["title"]):
                        deterministic = self._deterministic_summary(
                            title=child_title,
                            parent_title=block["title"],
                            full_content=child_content,
                        )
                        content_hash = self._hash_text(self._canonical_content(child_content))
                        old = old_by_hash.get(content_hash)
                        summary = deterministic
                        summary_source = "deterministic"
                        if self.summary_enabled and old and old.get("summary") and old.get("summary_source", "").startswith("llm"):
                            summary = old["summary"]
                            summary_source = old.get("summary_source") or "llm"
                            stats["summaries_llm_reused"] += 1
                        elif self.summary_enabled and self._summary_client_ready():
                            try:
                                summary = await self._summarize_with_llm(
                                    source_file=path.name,
                                    parent_title=block["title"],
                                    child_title=child_title,
                                    full_content=child_content,
                                )
                                summary_source = f"llm:{self._effective_summary_model()}"
                                stats["summaries_llm_created"] += 1
                            except Exception as exc:
                                logger.warning("LLM summary failed for %s / %s: %s", block["title"], child_title, exc)
                                stats["errors"].append(f"summary {path.name}/{block['title']}/{child_title}: {exc}")
                                stats["summaries_deterministic"] += 1
                        else:
                            stats["summaries_deterministic"] += 1

                        keywords = self._keywords_for(block["title"], child_title, child_content, summary)
                        elem_id = self._element_id(path.name, block["title"], child_title, content_hash)
                        parsed.append(IndexedGMElement(
                            id=elem_id,
                            source_file=path.name,
                            source_path=str(path),
                            parent_title=block["title"],
                            child_title=child_title,
                            scopes=block["scopes"],
                            full_content=child_content.strip(),
                            summary=self._clip_summary(summary),
                            deterministic_summary=deterministic,
                            content_hash=content_hash,
                            keywords=keywords,
                            order_index=order,
                            summary_source=summary_source,
                        ))
                        order += 1
            except Exception as exc:
                logger.warning("Static GM index could not parse %s: %s", path, exc)
                stats["errors"].append(f"{path}: {exc}")

        stats["gm_child_elements"] = len(parsed)
        stats["elapsed_seconds"] = round(time.time() - started, 3)
        await asyncio.to_thread(self._replace_db_rows, parsed, stats)
        self._stats.update(stats)
        return self.get_stats()

    def _old_summaries_by_hash(self) -> Dict[str, Dict[str, str]]:
        self._ensure_schema()
        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT content_hash, summary, summary_source FROM gm_elements").fetchall()
        out: Dict[str, Dict[str, str]] = {}
        for row in rows:
            out[str(row["content_hash"])] = {
                "summary": str(row["summary"] or ""),
                "summary_source": str(row["summary_source"] or ""),
            }
        return out

    def _replace_db_rows(self, elements: Sequence[IndexedGMElement], stats: Dict[str, Any]) -> None:
        self._ensure_schema()
        now = time.time()
        with sqlite3.connect(self.db_path) as db:
            db.execute("DELETE FROM gm_elements")
            db.executemany(
                """
                INSERT INTO gm_elements (
                    id, source_file, source_path, parent_title, child_title, scopes_json,
                    full_content, summary, deterministic_summary, content_hash, keywords_json,
                    order_index, summary_source, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        e.id,
                        e.source_file,
                        e.source_path,
                        e.parent_title,
                        e.child_title,
                        json.dumps(e.scopes, ensure_ascii=False),
                        e.full_content,
                        e.summary,
                        e.deterministic_summary,
                        e.content_hash,
                        json.dumps(e.keywords, ensure_ascii=False),
                        e.order_index,
                        e.summary_source,
                        now,
                    )
                    for e in elements
                ],
            )
            db.execute(
                "INSERT OR REPLACE INTO gm_index_meta(key, value) VALUES (?, ?)",
                ("last_reindex", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))),
            )
            db.execute(
                "INSERT OR REPLACE INTO gm_index_meta(key, value) VALUES (?, ?)",
                ("last_reindex_result", json.dumps(stats, ensure_ascii=False)),
            )
            db.commit()

    def _normalize_policy_name(self, policy: str) -> str:
        raw = str(policy or "").strip().lower()
        return "pinned" if raw == "pin" else raw

    def _strip_inline_policy_end(self, text: str) -> Tuple[str, bool]:
        value = str(text or "").strip()
        match = self.INLINE_POLICY_END_RE.search(value)
        if not match:
            return value, False
        return value[:match.start()].strip(), True

    def _parse_policy_blocks(self, text: str, path: Path) -> List[Dict[str, Any]]:
        matches = list(self.POLICY_HEADER_RE.finditer(text or ""))
        blocks: List[Dict[str, Any]] = []
        for idx, match in enumerate(matches):
            policy_raw = match.group("policy_eq") or match.group("policy_bare") or ""
            policy = self._normalize_policy_name(policy_raw)
            scope_blob = match.group("scopes_eq") if match.group("policy_eq") else match.group("scopes_bare")
            title_blob = match.group("title_eq") if match.group("policy_eq") else match.group("title_bare")
            title, inline_closed = self._strip_inline_policy_end(title_blob or "")
            scopes = self._parse_scopes(scope_blob or "")
            title = self._normalize_space(title or "Untitled") or "Untitled"
            body_start = match.end()
            next_policy_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            if inline_closed:
                body_end = body_start
            else:
                end_match = next(self.END_POLICY_RE.finditer(text, body_start, next_policy_start), None)
                body_end = end_match.start() if end_match else next_policy_start
            body = text[body_start:body_end]
            body = self.END_POLICY_RE.sub("", body).strip()
            blocks.append({
                "policy": policy,
                "scopes": scopes,
                "title": title,
                "body": body,
                "inline_closed": inline_closed,
                "source_file": path.name,
                "source_path": str(path),
            })
        return blocks

    def _split_gm_children(self, body: str, parent_title: str) -> List[Tuple[str, str]]:
        body = (body or "").strip()
        if not body:
            return []
        matches = list(self.CHILD_HEADER_RE.finditer(body))
        children: List[Tuple[str, str]] = []
        if not matches:
            visible_parent = f"=== {parent_title} ==="
            return [(parent_title, f"{visible_parent}\n{body}".strip())]
        for idx, match in enumerate(matches):
            title = self._normalize_space(match.group(2) or "Untitled") or "Untitled"
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
            content = body[start:end].strip()
            if content:
                children.append((title, content))
        return children

    def _deterministic_summary(self, *, title: str, parent_title: str, full_content: str) -> str:
        parts = []
        if parent_title:
            parts.append(f"Parent: {parent_title}.")
        if title:
            parts.append(f"Entry: {title}.")
        lines: List[str] = []
        for raw in str(full_content or "").splitlines():
            line = self._normalize_space(raw)
            if not line or line.startswith("#") or line.startswith("==="):
                continue
            if re.match(r'^[A-Za-z][A-Za-z0-9 /\'’"(),._-]{0,50}\s*:\s*.+$', line):
                lines.append(line)
            elif len(lines) < 4:
                lines.append(line)
        selected: List[str] = []
        used = 0
        for line in lines:
            clipped = line[:180].strip()
            projected = used + len(clipped) + 1
            if selected and projected > 900:
                break
            selected.append(clipped)
            used = projected
            if len(selected) >= 8:
                break
        parts.extend(selected)
        return self._clip_summary(" ".join(parts) or title or parent_title)

    async def _summarize_with_llm(self, *, source_file: str, parent_title: str, child_title: str, full_content: str) -> str:
        api_url = self._effective_summary_api_url().rstrip("/")
        if not api_url.endswith("/chat/completions"):
            api_url = f"{api_url}/chat/completions"
        model = self._effective_summary_model()
        if not api_url or not model:
            raise RuntimeError("summary API URL/model not configured")
        headers = {"Content-Type": "application/json"}
        key = self._effective_summary_api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        system = self.summary_instruction or DEFAULT_SUMMARY_INSTRUCTION
        clipped = str(full_content or "")[: self.summary_max_chars]
        user = (
            f"Source file: {source_file}\n"
            f"Parent GM section: {parent_title}\n"
            f"Child element: {child_title}\n\n"
            f"Full element text:\n{clipped}\n\n"
            "Write the summary now. Use only the full element text; the parent and child titles are metadata. "
            "Keep the wording dense and useful for later ID selection."
        )
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.summary_temperature,
            "max_tokens": self.summary_max_tokens,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.summary_timeout_seconds)) as client:
            response = await client.post(api_url, headers=headers, json=body)
        response.raise_for_status()
        payload = response.json()
        text = self._extract_response_text(payload)
        if not text:
            raise RuntimeError("summary model returned no text")
        return self._clip_summary(text)

    def _score_text(
        self,
        text: str,
        query_tokens: Sequence[str],
        entity_tokens: Iterable[str],
        entity_values: Sequence[str],
    ) -> Tuple[float, str]:
        haystack = self._normalize_for_lookup(text)
        if not haystack:
            return 0.0, "empty"
        score = 0.0
        reasons: List[str] = []
        token_hits = 0
        for token in query_tokens:
            if token and token in haystack:
                token_hits += 1
        if token_hits:
            score += min(2.5, token_hits * 0.18)
            reasons.append(f"query_tokens={token_hits}")
        ent_hits = 0
        for value in entity_values:
            norm = self._normalize_for_lookup(value)
            if norm and norm in haystack:
                ent_hits += 1
                score += 0.85
        if ent_hits:
            reasons.append(f"entities={ent_hits}")
        loose_entity_hits = 0
        for token in entity_tokens:
            if token and token in haystack:
                loose_entity_hits += 1
        if loose_entity_hits:
            score += min(1.5, loose_entity_hits * 0.16)
            reasons.append(f"entity_tokens={loose_entity_hits}")
        # Title-like exact words matter more than generic body overlap.
        if any(token in haystack[:300] for token in query_tokens):
            score += 0.25
            reasons.append("early_hit")
        return score, ",".join(reasons) or "fallback_order"

    def _should_reindex(self) -> bool:
        if not self.enabled:
            return False
        if not self.db_path.exists():
            return True
        if not self._memory:
            return True
        try:
            db_mtime = self.db_path.stat().st_mtime
            return any(path.exists() and path.stat().st_mtime > db_mtime for path in self._configured_paths())
        except Exception:
            return True

    def _configured_paths(self) -> List[Path]:
        if not self.ai_influence_folder:
            return []
        base = self._resolve_path(self.ai_influence_folder)
        return [base / file_name for file_name in self.files if str(file_name or "").strip()]

    def _resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        config_path = getattr(self.settings, "config_path", None)
        if config_path:
            return (Path(config_path).resolve().parent.parent / path).resolve()
        return path.resolve()

    def _summary_client_ready(self) -> bool:
        return bool(self._effective_summary_api_url() and self._effective_summary_model())

    def _effective_summary_api_url(self) -> str:
        return self.summary_api_url or str(getattr(self.settings, "selector_api_url", "") or getattr(self.settings, "api_url", "") or "").strip()

    def _effective_summary_api_key(self) -> str:
        return self.summary_api_key or str(getattr(self.settings, "selector_api_key", "") or getattr(self.settings, "api_key", "") or "")

    def _effective_summary_model(self) -> str:
        if self.summary_model:
            return self.summary_model
        selector_model = str(getattr(self.settings, "selector_model", "") or "").strip()
        if selector_model:
            return selector_model
        models = getattr(self.settings, "models", {}) or {}
        if isinstance(models, dict):
            for key in ("dialogue", "events", "diplomacy"):
                if str(models.get(key, "") or "").strip():
                    return str(models[key]).strip()
        return ""

    def _extract_response_text(self, payload: Any) -> str:
        try:
            choices = payload.get("choices") if isinstance(payload, dict) else None
            first = choices[0] if isinstance(choices, list) and choices else None
            message = first.get("message") if isinstance(first, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if isinstance(item.get("text"), str):
                            parts.append(item["text"])
                        elif isinstance(item.get("content"), str):
                            parts.append(item["content"])
                return "\n".join(parts).strip()
        except Exception:
            return ""
        return ""

    def _parse_scopes(self, scope_blob: str) -> List[str]:
        scopes = []
        for raw in str(scope_blob or "").split(":"):
            norm = re.sub(r'[^a-z0-9]+', '_', raw.lower()).strip('_')
            if norm and norm not in scopes:
                scopes.append(norm)
        return scopes

    def _keywords_for(self, parent: str, child: str, content: str, summary: str) -> List[str]:
        seen = set()
        keywords: List[str] = []
        for token in self._important_tokens("\n".join([parent, child, content, summary]), min_len=3):
            if token in seen:
                continue
            seen.add(token)
            keywords.append(token)
            if len(keywords) >= 80:
                break
        return keywords

    def _entity_values(self, entities: Dict[str, List[str]]) -> List[str]:
        values: List[str] = []
        seen = set()
        for bucket in ("kingdoms", "characters", "locations", "cultures", "string_ids"):
            for value in entities.get(bucket, []):
                cleaned = self._normalize_space(str(value or ""))
                norm = cleaned.lower()
                if cleaned and norm not in seen:
                    seen.add(norm)
                    values.append(cleaned)
        return values

    def _important_tokens(self, text: str, *, min_len: int = 4) -> List[str]:
        stop = {
            "that", "this", "with", "from", "into", "your", "their", "there", "where", "when", "what",
            "about", "should", "would", "could", "character", "current", "dialogue", "event", "events",
            "player", "message", "response", "generate", "write", "rules", "section", "context", "data",
        }
        tokens: List[str] = []
        seen = set()
        for match in self.TOKEN_RE.finditer(str(text or "").lower()):
            token = match.group(0).strip("_'-")
            if len(token) < min_len or token in stop or token.isdigit():
                continue
            if token in seen:
                continue
            seen.add(token)
            tokens.append(token)
            if len(tokens) >= 220:
                break
        return tokens

    def _element_id(self, source_file: str, parent: str, child: str, content_hash: str) -> str:
        return f"{self._slug(source_file)}:{self._slug(parent)}:{self._slug(child)}:{content_hash[:10]}"

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(str(text or "").encode("utf-8", errors="ignore")).hexdigest()

    def _canonical_content(self, text: str) -> str:
        cleaned = re.sub(r'\s+', ' ', str(text or "")).strip()
        return cleaned

    def _clip_summary(self, text: str) -> str:
        cleaned = self._normalize_space(text)
        if len(cleaned) > 1200:
            return cleaned[:1197].rstrip() + "..."
        return cleaned

    def _slug(self, value: str) -> str:
        slug = re.sub(r'[^a-z0-9]+', '_', str(value or "").lower()).strip('_')
        return slug or "untitled"

    def _normalize_space(self, text: str) -> str:
        return re.sub(r'\s+', ' ', str(text or "")).strip()

    def _normalize_for_lookup(self, text: str) -> str:
        return re.sub(r'[^a-z0-9_]+', ' ', str(text or "").lower()).strip()

    def _is_empty_section(self, section: Any) -> bool:
        for line in str(getattr(section, "full_content", "") or "").splitlines()[1:]:
            if line.strip():
                return False
        return True
