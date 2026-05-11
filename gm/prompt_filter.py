# Copyright (C) 2026 SEGAREGA
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Prompt Filter - Reconstruct prompts with GM-filtered content

Strategy:
1. Prompt-only interception: do NOT read or append configured files from disk.
2. Incoming prompt tags -> parse in place with [PINNED]/[GM]/[IGNORE]/[END ...]. Plain tags apply to all request types; scoped tags like [GM:EVENTS] apply only to matching request types.
3. Dynamic AIInfluence data -> filter known heavy sections by relevance.
4. Game state/event history/dialogue history embedded in the prompt -> preserve complete relevant objects/entries.
5. Dialogue history embedded in the prompt -> preserve the last N whole entries.

Important boundary:
- Dialogue requests usually have a system prompt plus a current user message. In that case only the system prompt is reconstructed/filtered; the current user message is left unchanged by main.py.
- Event/diplomacy requests may have no system prompt and instead put the whole prompt container into one user message. In that case main.py asks this class to filter that user message in place.
- No output text is manually character-truncated. Size control is done by selecting fewer complete GM sections/objects.
"""

import json
import hashlib
import logging
import re
from contextvars import ContextVar
from difflib import SequenceMatcher
from typing import Dict, List, Any, Optional, Iterable, Set
from dataclasses import dataclass, field

from .retriever import GMContentManager, Section, SelectedSection
from .selector import GMSelectorClient
from config.settings import Settings

logger = logging.getLogger(__name__)


def normalize_prompt_rule_text(value: Any) -> str:
    """Accept real marker text and GUI-pasted JSON-escaped marker text.

    Correct settings.json will contain JSON escapes such as \" in the file, but
    json.loads turns those into real quote characters. If a user pasted those
    escapes into the GUI and produced literal backslash-quote markers, normalize
    them at runtime so the exact-span rules still match the real prompt.
    """
    text = str(value or "")
    if '\"' not in text:
        return text
    try:
        decoded = json.loads('"' + text + '"')
    except Exception:
        return text
    return decoded if isinstance(decoded, str) else text


@dataclass
class FilteredPrompt:
    """Result of filtering a prompt."""
    system_prompt: str
    filtered_content: Dict[str, str]
    original_size: int
    filtered_size: int
    reduction_pct: float
    sections_included: int
    pinned_sections_included: int
    entities_found: Dict[str, List[str]]


@dataclass
class StaticGMSelectionState:
    """Request-local static GM selector state.

    The selector must run once against the indexed DB candidate set. Prompt-present
    GM sections are then filtered by the selected DB IDs, and DB-only selected
    entries are appended once after prompt filtering. This prevents the old
    per-block selector path from duplicating selected entries or leaving an
    unfiltered copy of the original file content in the main prompt.
    """
    selected_ids: Set[str] = field(default_factory=set)
    selected_order: List[str] = field(default_factory=list)
    selected_sections_by_id: Dict[str, Any] = field(default_factory=dict)
    rendered_ids: Set[str] = field(default_factory=set)
    prompt_seen_ids: Set[str] = field(default_factory=set)
    selection_done: bool = False
    selector_candidates: int = 0
    candidate_parent_counts: Dict[str, int] = field(default_factory=dict)


_STATIC_GM_SELECTION_STATE: ContextVar[Optional[StaticGMSelectionState]] = ContextVar(
    "static_gm_selection_state", default=None
)


class PromptFilter:
    """Filter and reconstruct whichever prompt-container message main.py selects."""

    # Sections that are data-heavy and should be replaced by the filtered version.
    BULK_SECTION_ALIASES = {
        'world lore', 'world_lore', 'world', 'lore', 'world txt', 'world.txt',
        'game state', 'game_state', 'gamestate', 'current game state',
        'event history', 'event_history', 'recent events', 'events',
        'dialogue history', 'dialogue_history', 'conversation history', 'conversation_history',
        'recent dialogue',
        'cultural traditions', 'cultural_traditions', 'cultural traditions json', 'cultural_traditions.json',
        'rules', 'statement rules', 'event rules', 'analyzer rules', 'action rules', 'player description',
        'kingdom statement rules', 'events generator rules', 'events analyzer rules',
        'kingdomstatementrules', 'eventsgeneratorrules', 'eventsanalyzerrules', 'actionrules', 'playerdescription',
        'rules_action_rules', 'rules_statement_rules', 'rules_event_rules', 'rules_analyzer_rules',
        'rules_player_description', 'rules_kingdom_statement_rules', 'rules_events_generator_rules',
        'rules_events_analyzer_rules',
        # AIInfluence prompt-container sections that are reconstructed by this proxy.
        'the world', 'global politics of the world',
        'character briefing current data', 'immediate situation current data', 'the player current data',
        'nearby settlements strategic context current data', 'nearby parties npc vicinity current data',
        'people physically present in this location right now',
        'mentioned settlements', 'mentioned characters', 'mentioned parties',
        'current world data ground truth', 'current world state',
        'existing events historical narrative', 'existing dynamic events do not duplicate',
        'active economic effects', 'recent diplomatic statements last 15 statements from last 50 days',
        'new npc dialogues since last analysis',
        # World/lore blocks seen in user-only event/diplomacy prompts.
        'world information', 'economic economy and prices', 'factions civilizations and factions',
        'politics political power ecology', 'religion faiths', 'geography world geography',
        'education education system', 'slavery slavery',
    }

    def __init__(self, retriever: GMContentManager, settings: Settings, selector_client: Optional[GMSelectorClient] = None):
        self.retriever = retriever
        self.settings = settings
        self.selector_client = selector_client
        self.filtering_mode = settings.get_filtering_mode()

        # Token estimation (rough: 1 token ≈ 4 chars)
        self.CHARS_PER_TOKEN = 4

        # Wired to settings.json.
        self.MAX_EVENT_HISTORY = settings.max_event_history
        self.DIALOGUE_HISTORY_SIZE = settings.dialogue_history_size
        self.dynamic_filter_enabled = getattr(settings, "dynamic_filter_enabled", True)
        self.fuzzy_match_threshold = float(getattr(settings, "fuzzy_match_threshold", 0.88))
        self.max_people_present = int(getattr(settings, "max_people_present", 8))
        self.max_nearby_settlements = int(getattr(settings, "max_nearby_settlements", 8))
        self.max_nearby_parties = int(getattr(settings, "max_nearby_parties", 8))
        self.max_inventory_lines = int(getattr(settings, "max_inventory_lines", 8))
        # Plain [GM] applies to all request types. Scoped tags such as
        # [GM:EVENTS] and [GM:DIPLOMACY] are request-specific.
        self.max_event_dialogue_messages = int(getattr(settings, "max_event_dialogue_messages", 14))
        self.max_event_dialogue_settlements = int(getattr(settings, "max_event_dialogue_settlements", 10))
        self.prompt_drop_rules = list(getattr(settings, "prompt_drop_rules", []) or [])
        self.prompt_replace_rules = list(getattr(settings, "prompt_replace_rules", []) or [])
        self.selector_context_rules = list(getattr(settings, "selector_context_rules", []) or [])
        self.static_gm_index_enabled = bool(getattr(settings, "static_gm_index_enabled", False))
        self.static_gm_index_selector_payload = str(getattr(settings, "static_gm_index_selector_payload", "summary") or "summary").strip().lower()

    def _normalize_rule_type(self, value: Any) -> Optional[str]:
        """Normalize request-type names used by prompt drop/replace rules."""
        if value is None:
            return None
        raw = str(value).strip().lower()
        if not raw:
            return None
        aliases = {
            "chat": "dialogue",
            "dialog": "dialogue",
            "dialogue": "dialogue",
            "event": "events",
            "events": "events",
            "event_generation": "events",
            "kingdom_statement": "diplomacy",
            "kingdomstatement": "diplomacy",
            "statement": "diplomacy",
            "statements": "diplomacy",
            "diplomacy": "diplomacy",
            "diplomatic": "diplomacy",
            "all": "all",
            "any": "all",
            "*": "all",
        }
        return aliases.get(raw, raw)

    def _rule_applies_to_request_type(self, rule: Dict[str, Any], request_type: str, rule_kind: str, name: str) -> bool:
        """Return True if a prompt drop/replace rule should run for this request type.

        The optional setting is named `request_types` for readability. `type` and `types` are also accepted as aliases.
        If omitted, the rule applies to dialogue, events, diplomacy, and unknown requests.
        """
        raw_types = rule.get("request_types", rule.get("type", rule.get("types", None)))
        if raw_types is None:
            return True

        if isinstance(raw_types, str):
            requested = [part for part in re.split(r"[,|]", raw_types) if part.strip()]
        elif isinstance(raw_types, (list, tuple, set)):
            requested = list(raw_types)
        else:
            logger.warning("Prompt %s rule %s skipped: request_types/type/types must be a string or list", rule_kind, name)
            return False

        allowed = {self._normalize_rule_type(item) for item in requested}
        allowed.discard(None)
        if not allowed:
            logger.warning("Prompt %s rule %s skipped: request_types/type/types did not contain a valid request type", rule_kind, name)
            return False
        if "all" in allowed:
            return True

        current = self._normalize_rule_type(request_type) or "unknown"
        if current not in allowed:
            logger.info("Prompt %s rule %s skipped for request_type=%s; allowed=%s", rule_kind, name, current, sorted(allowed))
            return False
        return True

    def _apply_prompt_drop_rules(self, text: str, request_type: str) -> str:
        """Apply user-configured exact-span drop rules to the intercepted prompt container.

        Rules are deliberately fail-closed: if a configured start/end/scope marker cannot be
        found, the rule is skipped. This prevents a typo from deleting the rest of a prompt.
        Rules may optionally include `type` or `types` to run only for dialogue/events/diplomacy.
        """
        if not text or not self.prompt_drop_rules:
            return text

        result = text
        for idx, rule in enumerate(self.prompt_drop_rules):
            if not isinstance(rule, dict):
                logger.warning("Prompt drop rule %d skipped: rule is not an object", idx)
                continue
            name = str(rule.get("name") or f"drop_rule_{idx}")
            if not self._rule_applies_to_request_type(rule, request_type, "drop", name):
                continue
            result = self._apply_single_prompt_drop_rule(result, rule, idx)
        return result

    def _apply_prompt_replace_rules(self, text: str, request_type: str) -> str:
        """Apply user-configured exact-span replace rules to the intercepted prompt container.

        Replace rules are deliberately fail-closed: if a configured start/end/scope
        marker cannot be found, the rule is skipped. They preserve the original prompt
        order and replace only the matched span in place. Rules may optionally include
        `type` or `types` to run only for dialogue/events/diplomacy.
        """
        if not text or not self.prompt_replace_rules:
            return text

        result = text
        for idx, rule in enumerate(self.prompt_replace_rules):
            if not isinstance(rule, dict):
                logger.warning("Prompt replace rule %d skipped: rule is not an object", idx)
                continue
            name = str(rule.get("name") or f"replace_rule_{idx}")
            if not self._rule_applies_to_request_type(rule, request_type, "replace", name):
                continue
            result = self._apply_single_prompt_replace_rule(result, rule, idx)
        return result

    def _apply_single_prompt_replace_rule(self, text: str, rule: Dict[str, Any], idx: int) -> str:
        name = str(rule.get("name") or f"replace_rule_{idx}")
        replace_beginning = rule.get("replace_beginning")
        replace_end = rule.get("replace_end")
        replacement_text = rule.get("replacement_text")
        if isinstance(replace_beginning, str):
            replace_beginning = normalize_prompt_rule_text(replace_beginning)
        if isinstance(replace_end, str):
            replace_end = normalize_prompt_rule_text(replace_end)
        if isinstance(replacement_text, str):
            replacement_text = normalize_prompt_rule_text(replacement_text)

        if not isinstance(replace_beginning, str) or not replace_beginning:
            logger.warning("Prompt replace rule %s skipped: missing replace_beginning", name)
            return text
        if not isinstance(replace_end, str) or not replace_end:
            logger.warning("Prompt replace rule %s skipped: missing replace_end", name)
            return text
        if not isinstance(replacement_text, str):
            logger.warning("Prompt replace rule %s skipped: missing replacement_text", name)
            return text

        search_start = 0
        search_end = len(text)

        scope_beginning = rule.get("scope_beginning")
        scope_end = rule.get("scope_end")
        if isinstance(scope_beginning, str):
            scope_beginning = normalize_prompt_rule_text(scope_beginning)
        if isinstance(scope_end, str):
            scope_end = normalize_prompt_rule_text(scope_end)
        if isinstance(scope_beginning, str) and scope_beginning:
            scope_start_idx = text.find(scope_beginning)
            if scope_start_idx == -1:
                logger.warning("Prompt replace rule %s skipped: scope_beginning not found", name)
                return text
            search_start = scope_start_idx
            if isinstance(scope_end, str) and scope_end:
                scope_end_idx = text.find(scope_end, search_start + len(scope_beginning))
                if scope_end_idx == -1:
                    logger.warning("Prompt replace rule %s skipped: scope_end not found after scope_beginning", name)
                    return text
                search_end = scope_end_idx
        elif isinstance(scope_end, str) and scope_end:
            scope_end_idx = text.find(scope_end)
            if scope_end_idx == -1:
                logger.warning("Prompt replace rule %s skipped: scope_end not found", name)
                return text
            search_end = scope_end_idx

        begin_idx = text.find(replace_beginning, search_start, search_end)
        if begin_idx == -1:
            logger.warning("Prompt replace rule %s skipped: replace_beginning not found", name)
            return text

        end_idx = text.find(replace_end, begin_idx + len(replace_beginning), search_end)
        if end_idx == -1:
            logger.warning("Prompt replace rule %s skipped: replace_end not found after replace_beginning", name)
            return text

        # Marker-based span behavior: these booleans control whether the exact
        # configured beginning/end marker strings are included in the replaced span.
        # They do not expand to whole physical lines.
        delete_replace_beginning_marker = bool(rule.get("delete_replace_beginning_marker", True))
        delete_replace_end_marker = bool(rule.get("delete_replace_end_marker", True))

        replace_start = begin_idx if delete_replace_beginning_marker else begin_idx + len(replace_beginning)
        replace_end_pos = end_idx + len(replace_end) if delete_replace_end_marker else end_idx

        if replace_end_pos < replace_start:
            logger.warning("Prompt replace rule %s skipped: computed invalid replacement span", name)
            return text

        replacement = replacement_text

        replaced_chars = replace_end_pos - replace_start
        logger.info("Prompt replace rule %s replaced %d chars with %d chars", name, replaced_chars, len(replacement))
        return text[:replace_start] + replacement + text[replace_end_pos:]

    def _apply_single_prompt_drop_rule(self, text: str, rule: Dict[str, Any], idx: int) -> str:
        name = str(rule.get("name") or f"drop_rule_{idx}")
        drop_beginning = rule.get("drop_beginning")
        drop_end = rule.get("drop_end")
        if isinstance(drop_beginning, str):
            drop_beginning = normalize_prompt_rule_text(drop_beginning)
        if isinstance(drop_end, str):
            drop_end = normalize_prompt_rule_text(drop_end)
        if not isinstance(drop_beginning, str) or not drop_beginning:
            logger.warning("Prompt drop rule %s skipped: missing drop_beginning", name)
            return text
        if not isinstance(drop_end, str) or not drop_end:
            logger.warning("Prompt drop rule %s skipped: missing drop_end", name)
            return text

        search_start = 0
        search_end = len(text)

        scope_beginning = rule.get("scope_beginning")
        scope_end = rule.get("scope_end")
        if isinstance(scope_beginning, str):
            scope_beginning = normalize_prompt_rule_text(scope_beginning)
        if isinstance(scope_end, str):
            scope_end = normalize_prompt_rule_text(scope_end)
        if isinstance(scope_beginning, str) and scope_beginning:
            scope_start_idx = text.find(scope_beginning)
            if scope_start_idx == -1:
                logger.warning("Prompt drop rule %s skipped: scope_beginning not found", name)
                return text
            search_start = scope_start_idx
            if isinstance(scope_end, str) and scope_end:
                scope_end_idx = text.find(scope_end, search_start + len(scope_beginning))
                if scope_end_idx == -1:
                    logger.warning("Prompt drop rule %s skipped: scope_end not found after scope_beginning", name)
                    return text
                search_end = scope_end_idx
        elif isinstance(scope_end, str) and scope_end:
            scope_end_idx = text.find(scope_end)
            if scope_end_idx == -1:
                logger.warning("Prompt drop rule %s skipped: scope_end not found", name)
                return text
            search_end = scope_end_idx

        begin_idx = text.find(drop_beginning, search_start, search_end)
        if begin_idx == -1:
            logger.warning("Prompt drop rule %s skipped: drop_beginning not found", name)
            return text

        end_idx = text.find(drop_end, begin_idx + len(drop_beginning), search_end)
        if end_idx == -1:
            logger.warning("Prompt drop rule %s skipped: drop_end not found after drop_beginning", name)
            return text

        # Marker-based span behavior: these booleans control whether the exact
        # configured beginning/end marker strings are included in the deleted span.
        # They do not expand to whole physical lines.
        delete_drop_beginning_marker = bool(rule.get("delete_drop_beginning_marker", True))
        delete_drop_end_marker = bool(rule.get("delete_drop_end_marker", False))

        delete_start = begin_idx if delete_drop_beginning_marker else begin_idx + len(drop_beginning)
        delete_end = end_idx + len(drop_end) if delete_drop_end_marker else end_idx

        if delete_end < delete_start:
            logger.warning("Prompt drop rule %s skipped: computed invalid deletion span", name)
            return text

        removed_chars = delete_end - delete_start
        logger.info("Prompt drop rule %s removed %d chars", name, removed_chars)
        return text[:delete_start] + text[delete_end:]

    def _line_bounds(self, text: str, index: int) -> tuple[int, int]:
        """Return (line_start, line_end_including_newline_if_present) for an index."""
        line_start = text.rfind('\n', 0, index) + 1
        newline_idx = text.find('\n', index)
        if newline_idx == -1:
            return line_start, len(text)
        return line_start, newline_idx + 1

    async def filter_prompt(
        self,
        messages: List[Dict[str, Any]],
        request_type: str = "dialogue",
        target_message_index: Optional[int] = None,
    ) -> FilteredPrompt:
        """
        Filter one prompt-container message IN PLACE.

        v11 intentionally does not rebuild the prompt into proxy-owned groups such as
        AI_CURRENT_DATA / WORLD_LORE. It preserves the original AIInfluence prompt order
        and only replaces/removes the spans that the proxy actually filters.
        """
        source_content = self._extract_target_content(messages, target_message_index)
        context_content = self._extract_context_content(messages, target_message_index)
        dialogue_query = self._extract_last_user_message(messages) if request_type == "dialogue" else ""
        original_size = len(source_content)

        filtered_text, sections_included, pinned_section_count, entities = await self._filter_prompt_in_place(
            source_content=source_content,
            context_content=context_content,
            dialogue_query=dialogue_query,
            request_type=request_type,
        )

        filtered_size = len(filtered_text)
        reduction = ((original_size - filtered_size) / original_size * 100) if original_size > 0 else 0.0

        logger.info(
            "Prompt-container filtered in place: %d -> %d chars (%.1f%% reduction), world_sections=%d",
            original_size,
            filtered_size,
            reduction,
            sections_included,
        )
        logger.debug(f"Entities found: {entities}")

        return FilteredPrompt(
            system_prompt=filtered_text,
            filtered_content={},
            original_size=original_size,
            filtered_size=filtered_size,
            reduction_pct=reduction,
            sections_included=sections_included,
            pinned_sections_included=pinned_section_count,
            entities_found=entities,
        )

    async def _filter_prompt_in_place(
        self,
        source_content: str,
        context_content: str,
        dialogue_query: str,
        request_type: str,
    ) -> tuple[str, int, int, Dict[str, List[str]]]:
        """Return an in-place filtered prompt and stats. No proxy wrappers are inserted."""
        state_token = None
        if self._static_gm_index_available():
            state_token = _STATIC_GM_SELECTION_STATE.set(StaticGMSelectionState())

        try:
            working = self._apply_prompt_drop_rules(source_content, request_type)
            working = self._apply_prompt_replace_rules(working, request_type)
            working = self._repair_split_policy_headers(working)

            selection_query = self._build_selection_query(
                context_content=context_content,
                dialogue_query=dialogue_query,
                source_content=working,
                request_type=request_type,
                game_state=None,
                event_history=None,
                dialogue_history=None,
                ai_relevance_signal='',
            )
            entities = self._extract_entities_from_text(selection_query)
            focus_text = self._focus_text(context_content, dialogue_query, working, request_type)
            selector_context = self._build_selector_context_segments(
                source_content=working,
                context_content=context_content,
                dialogue_query=dialogue_query,
                request_type=request_type,
            )

            # Some AIInfluence event/diplomacy prompts inject world.txt directly inside a
            # sentence, e.g. "You are operating in the world of === [PINNED] ... ==="
            # rather than under a clean "### The World ###" heading. Filter those explicit
            # tagged spans in place before the line-based section scanner runs. This does not
            # read files, insert wrappers, or reorder prompt sections.
            working, inline_selected, inline_pinned = await self._filter_inline_policy_regions_in_place(
                working,
                selection_query=selection_query,
                entities=entities,
                request_type=request_type,
                selector_context=selector_context,
                dialogue_query=dialogue_query,
            )

            filtered, sections_included, pinned_count = await self._replace_known_prompt_spans_in_place(
                text=working,
                focus_text=focus_text,
                selection_query=selection_query,
                entities=entities,
                request_type=request_type,
                context_content=context_content,
                selector_context=selector_context,
                dialogue_query=dialogue_query,
            )

            # DB-only selected GM entries have no original location in the intercepted prompt.
            # Append them once, after all prompt-present GM regions have had a chance to render
            # in place. Use only authored/stored parent headers; do not add an app-invented wrapper title.
            extra_static, extra_count = self._render_unrendered_selected_static_gm_sections()
            if extra_static:
                filtered = self._normalize_render_spacing(
                    f"{filtered.strip()}\n\n{extra_static}"
                )
                sections_included += extra_count

            filtered = self._repair_split_policy_headers(filtered)
            filtered = self._normalize_policy_header_seams(filtered)
            filtered = self._finalize_visible_policy_headers(filtered)
            return filtered.strip(), sections_included + inline_selected, pinned_count + inline_pinned, entities
        finally:
            if state_token is not None:
                _STATIC_GM_SELECTION_STATE.reset(state_token)


    def _is_policy_header_line(self, line: str) -> bool:
        """True for standalone policy headers.

        Accepted forms:
            === [GM:DIALOGUE] Cultures ===
            [GM:DIALOGUE] -- Cultures --
            [PINNED] Always remember this [END PIN]
        For compact headers, everything after the marker is treated as the human title/content.
        """
        stripped = str(line or "").strip()
        return bool(re.match(
            r'^(?:={3,}\s*\[(?:PINNED|PIN|GM|IGNORE)(?::[A-Z0-9_ -]+)*\]\s*.*?\s*={3,}|\[(?:PINNED|PIN|GM|IGNORE)(?::[A-Z0-9_ -]+)*\]\s*.*?)\s*$',
            stripped,
            re.IGNORECASE,
        ))

    def _line_has_inline_policy_end(self, line: str) -> bool:
        return bool(re.search(
            r'\[END\s+(?:PINNED|PIN|GM|IGNORE)(?::[A-Z0-9_ -]+)*\]',
            str(line or ""),
            re.IGNORECASE,
        ))

    def _policy_header_match_anywhere(self, text: str, start: int = 0):
        """Find an explicit policy header marker anywhere in text.

        This is used only for inline markers that AIInfluence embeds mid-line, for example:
        "You are operating in the world of === [PINNED] Game World ===".
        Standalone policy headers are handled by the line-based scanner so we do not double-filter them.
        """
        pattern = re.compile(
            r'={3,}\s*\[(?:PINNED|PIN|GM|IGNORE)(?::[A-Z0-9_ -]+)*\]\s*.*?\s*={3,}',
            re.IGNORECASE | re.DOTALL,
        )
        return pattern.search(text, start)

    def _policy_end_match_anywhere(self, text: str, start: int = 0):
        pattern = re.compile(
            r'={3,}\s*\[END\s+(?:PINNED|PIN|GM|IGNORE)(?::[A-Z0-9_ -]+)*\]\s*={3,}\s*\.?',
            re.IGNORECASE,
        )
        return pattern.search(text, start)

    def _ai_section_header_match_anywhere(self, text: str, start: int = 0):
        """Find the next AIInfluence-style section header in raw text."""
        pattern = re.compile(
            r'(?m)^\s*(?:#{3,6}\s+.+?\s*|={3,}\s*[^=\s].*?\s*={3,}\s*)$'
        )
        return pattern.search(text, start)

    def _inline_policy_block_end(self, text: str, start: int = 0):
        """Resolve the end of a policy block that starts mid-line.

        Prefer an explicit END marker when present. Otherwise, treat the next explicit
        policy header or next AIInfluence section header as the boundary. This matches
        the authored document rule the user wants: a policy block runs until the next
        policy header or outer prompt section, even when a newline was accidentally lost.
        """
        explicit_end = self._policy_end_match_anywhere(text, start)
        next_policy = self._policy_header_match_anywhere(text, start)
        next_ai = self._ai_section_header_match_anywhere(text, start)

        next_boundary_start = None
        for match in (next_policy, next_ai):
            if not match:
                continue
            if next_boundary_start is None or match.start() < next_boundary_start:
                next_boundary_start = match.start()

        if explicit_end and (next_boundary_start is None or explicit_end.start() <= next_boundary_start):
            return explicit_end.end()
        if next_boundary_start is not None:
            return next_boundary_start
        return len(text)

    async def _filter_inline_policy_regions_in_place(
        self,
        text: str,
        selection_query: str,
        entities: Dict[str, List[str]],
        request_type: str,
        selector_context: List[tuple[str, str]],
        dialogue_query: str,
    ) -> tuple[str, int, int]:
        """Filter explicit policy-tagged regions that begin mid-line.

        This fixes event/diplomacy prompts where AIInfluence embeds world.txt inline instead of
        under ### The World ###. It only processes regions with an explicit === [END ...] === marker
        so a missing marker cannot accidentally consume the rest of the prompt.
        """
        if not text:
            return text, 0, 0

        out: List[str] = []
        pos = 0
        total_selected = 0
        total_pinned = 0
        while True:
            m = self._policy_header_match_anywhere(text, pos)
            if not m:
                out.append(text[pos:])
                break

            # If the marker starts at the beginning of a physical line, leave it for the line-based
            # scanner; otherwise this is an inline marker that must be handled here.
            at_line_start = (m.start() == 0 or text[m.start() - 1] == '\n')
            if at_line_start:
                out.append(text[pos:m.end()])
                pos = m.end()
                continue

            end_pos = self._inline_policy_block_end(text, m.end())
            if end_pos <= m.start():
                logger.warning("Inline policy region skipped: could not resolve block boundary")
                out.append(text[pos:m.end()])
                pos = m.end()
                continue

            out.append(text[pos:m.start()])
            raw = text[m.start():end_pos]
            filtered, selected, pinned = await self._filter_tagged_policy_block_preserve_order(
                raw,
                selection_query=selection_query,
                entities=entities,
                request_type=request_type,
                selector_context=selector_context,
                dialogue_query=dialogue_query,
            )
            if filtered:
                if out and out[-1] and not out[-1].endswith(('\n', '\r')):
                    out.append('\n\n')
                out.append(filtered)
                if end_pos < len(text) and text[end_pos] not in '\r\n' and not filtered.endswith(('\n', '\r')):
                    out.append('\n\n')
            total_selected += selected
            total_pinned += pinned
            pos = end_pos

        return ''.join(out), total_selected, total_pinned

    async def _filter_tagged_policy_block_preserve_order(
        self,
        block: str,
        selection_query: str,
        entities: Dict[str, List[str]],
        request_type: str,
        selector_context: List[tuple[str, str]],
        dialogue_query: str,
    ) -> tuple[str, int, int]:
        """Apply [PINNED]/[GM]/[IGNORE] to an explicit tagged block in place.

        The output keeps the original document order. Pinned sections are included where they were.
        GM sections are included only if selected. Ignore/non-matching scoped sections are removed.
        """
        body = block.strip()
        if not body:
            return '', 0, 0

        body = self._repair_split_policy_headers(body)
        body = self._normalize_policy_block_boundaries(body)

        sections = self.retriever._parse_sections(body, 'incoming_policy_block', request_type=request_type)
        gm_sections = []
        pinned_count = 0
        for sec in sections:
            if sec.policy == self.retriever.POLICY_IGNORE:
                continue
            is_empty = self.retriever._is_empty_container_section(sec)
            if is_empty and not (self._selection_backend_enabled() and sec.policy == self.retriever.POLICY_GM and sec.level == 1):
                continue
            sec.entities = self.retriever._extract_entities(sec.full_content)
            sec.summary = self.retriever._generate_summary(sec)
            if sec.policy == self.retriever.POLICY_PINNED:
                pinned_count += 1
            else:
                gm_sections.append(sec)

        selected = await self._select_policy_gm_sections_for_request(
            gm_sections,
            selection_query,
            entities,
            request_type,
            selector_context=selector_context,
            dialogue_query=dialogue_query,
            block_title=sections[0].title if sections else "",
        )
        selected_ids = {id(sec) for sec in selected}
        rendered, selected_count = self._render_selected_world_sections(sections, selected_ids)
        return self._normalize_render_spacing('\n\n'.join(rendered)), selected_count, pinned_count

    def _current_static_selection_state(self) -> Optional[StaticGMSelectionState]:
        return _STATIC_GM_SELECTION_STATE.get()

    def _static_index_for_prompt_section(self, section: Any, parent_title: str = "") -> Optional[Any]:
        static_index = getattr(self.retriever, "static_gm_index", None)
        if static_index is None or not hasattr(static_index, "find_for_prompt_section"):
            return None
        try:
            return static_index.find_for_prompt_section(section, parent_title=parent_title)
        except Exception as exc:
            logger.debug("Static GM index lookup failed for prompt section %s/%s: %s", parent_title, getattr(section, "title", ""), exc)
            return None

    def _annotate_prompt_sections_with_static_ids(self, sections: List[Any]) -> tuple[List[Any], int]:
        """Attach static DB IDs to prompt-present GM sections and record which IDs exist in the prompt."""
        state = self._current_static_selection_state()
        parent_title = ""
        skipped_empty = 0
        annotated: List[Any] = []

        for section in sections:
            level = int(getattr(section, "level", 0) or 0)
            if level == 1:
                parent_title = str(getattr(section, "title", "") or "").strip()
                if self.retriever._is_empty_container_section(section):
                    skipped_empty += 1
                    continue

            if self.retriever._is_empty_container_section(section):
                skipped_empty += 1
                continue

            indexed = self._static_index_for_prompt_section(section, parent_title=parent_title)
            db_parent_title = ""
            if indexed is not None:
                db_parent_title = str(getattr(indexed, "parent_title", "") or "").strip()
                elem_id = str(getattr(indexed, "id", "") or "").strip()
                if elem_id:
                    setattr(section, "static_index_id", elem_id)
                    setattr(section, "static_index_source", str(getattr(indexed, "source_file", "") or ""))
                    setattr(section, "static_index_summary_source", str(getattr(indexed, "summary_source", "") or ""))
                    setattr(section, "static_index_parent_title", db_parent_title)
                    if state is not None:
                        state.prompt_seen_ids.add(elem_id)
                summary = str(getattr(indexed, "summary", "") or "").strip()
                if summary:
                    setattr(section, "selector_summary", summary)
            if not str(getattr(section, "selector_summary", "") or "").strip():
                setattr(section, "selector_summary", str(getattr(section, "summary", "") or self.retriever._generate_summary(section)))
            # For final rendering, prefer the DB parent title.  The live prompt parser can
            # occasionally see separator junk such as "#" as the current parent after
            # AIInfluence has reshaped the file, but the DB row stores the parent that was
            # indexed from the configured text file.
            setattr(section, "selector_parent_title", db_parent_title or parent_title)
            setattr(section, "external_static_gm", False)
            annotated.append(section)

        return annotated, skipped_empty

    def _indexed_element_to_selector_section(self, elem: Any) -> Any:
        """Convert a SQLite indexed GM row to a selector-only Section object."""
        parent = str(getattr(elem, "parent_title", "") or "").strip()
        child = str(getattr(elem, "child_title", "") or "").strip()
        source = str(getattr(elem, "source_file", "") or "").strip()
        summary = str(getattr(elem, "summary", "") or getattr(elem, "deterministic_summary", "") or "").strip()
        pseudo = Section(
            title=child or parent or "Indexed GM Entry",
            full_content=str(getattr(elem, "full_content", "") or "").strip(),
            summary=summary,
            source=source or "static_gm_index",
            level=2,
            policy=self.retriever.POLICY_GM,
            explicit_policy=self.retriever.POLICY_GM,
        )
        setattr(pseudo, "selector_summary", summary or self.retriever._generate_summary(pseudo))
        setattr(pseudo, "selector_parent_title", parent or source or "Indexed GM")
        setattr(pseudo, "static_index_parent_title", parent)
        setattr(pseudo, "static_index_id", str(getattr(elem, "id", "") or "").strip())
        setattr(pseudo, "static_index_source", source)
        setattr(pseudo, "static_index_summary_source", str(getattr(elem, "summary_source", "") or ""))
        setattr(pseudo, "external_static_gm", True)
        return pseudo

    def _all_indexed_static_gm_selector_sections(self, request_type: str) -> tuple[List[Any], Dict[str, int]]:
        """Return every indexed [GM] child that applies to this request type."""
        static_index = getattr(self.retriever, "static_gm_index", None)
        if static_index is None or not hasattr(static_index, "elements_for_request"):
            return [], {}
        try:
            elements = static_index.elements_for_request(request_type)
        except Exception as exc:
            logger.warning("Static GM index candidate enumeration failed: %s", exc)
            return [], {}

        sections: List[Any] = []
        parent_counts: Dict[str, int] = {}
        seen_ids: Set[str] = set()
        for elem in elements:
            elem_id = str(getattr(elem, "id", "") or "").strip()
            if not elem_id or elem_id in seen_ids:
                continue
            if not str(getattr(elem, "full_content", "") or "").strip():
                continue
            section = self._indexed_element_to_selector_section(elem)
            sections.append(section)
            seen_ids.add(elem_id)
            parent = str(getattr(section, "static_index_parent_title", "") or getattr(section, "selector_parent_title", "") or "Indexed GM").strip() or "Indexed GM"
            parent_counts[parent] = parent_counts.get(parent, 0) + 1
        return sections, parent_counts

    async def _select_static_index_gm_sections_for_request(
        self,
        sections: List[Any],
        query: str,
        entities: Dict[str, List[str]],
        request_type: str,
        selector_context: Optional[List[tuple[str, str]]] = None,
        dialogue_query: str = "",
        block_title: str = "",
    ) -> List[Any]:
        """Use the DB as the authoritative [GM] selector candidate source.

        The selector sees all indexed DB children once per prompt-filtering run. This method
        returns only the prompt-present sections whose DB IDs were explicitly selected. DB-only
        selected entries are appended once at the end of the filter pass.
        """
        state = self._current_static_selection_state()
        if state is None:
            state = StaticGMSelectionState()
            _STATIC_GM_SELECTION_STATE.set(state)

        prompt_sections, skipped_empty = self._annotate_prompt_sections_with_static_ids(sections)

        if not state.selection_done:
            candidate_sections, parent_counts = self._all_indexed_static_gm_selector_sections(request_type)
            state.selector_candidates = len(candidate_sections)
            state.candidate_parent_counts = dict(parent_counts)

            if self._selector_enabled() and self.selector_client:
                await self.selector_client.log_diagnostic(
                    "STATIC GM INDEX CANDIDATES",
                    {
                        "request_type": request_type,
                        "selector_candidates": len(candidate_sections),
                        "selector_payload": self.static_gm_index_selector_payload,
                        "local_shortlist": False,
                        "selection_mode": "all_indexed_db_gm_candidates_once",
                        "prompt_present_gm_seen_this_block": len(prompt_sections),
                        "empty_container_sections_skipped_this_block": skipped_empty,
                        "candidate_parent_counts": parent_counts,
                    },
                )

            selected_candidates = await self._select_via_selector_model(
                sections=candidate_sections,
                request_type=request_type,
                dialogue_query=dialogue_query,
                context_segments=selector_context or [],
                block_title=block_title,
                fallback_query=query,
                fallback_entities=entities,
            ) if self._selector_enabled() else []

            selected_ids: List[str] = []
            for section in selected_candidates:
                sid = str(getattr(section, "static_index_id", "") or "").strip()
                if not sid or sid in state.selected_ids:
                    continue
                state.selected_ids.add(sid)
                selected_ids.append(sid)
                state.selected_sections_by_id[sid] = section
            state.selected_order = selected_ids
            state.selection_done = True

            if self.selector_client:
                selected_parent_counts: Dict[str, int] = {}
                for sid in state.selected_order:
                    section = state.selected_sections_by_id.get(sid)
                    if not section:
                        continue
                    parent = str(getattr(section, "static_index_parent_title", "") or getattr(section, "selector_parent_title", "") or "Indexed GM").strip() or "Indexed GM"
                    selected_parent_counts[parent] = selected_parent_counts.get(parent, 0) + 1
                await self.selector_client.log_diagnostic(
                    "STATIC GM SELECTED FOR MAIN PROMPT",
                    {
                        "request_type": request_type,
                        "selector_candidates": state.selector_candidates,
                        "selected_total": len(state.selected_ids),
                        "selected_parent_counts": selected_parent_counts,
                    },
                )

        selected_prompt_sections: List[Any] = []
        for section in prompt_sections:
            sid = str(getattr(section, "static_index_id", "") or "").strip()
            if sid and sid in state.selected_ids:
                selected_prompt_sections.append(section)
        return selected_prompt_sections

    def _render_unrendered_selected_static_gm_sections(self) -> tuple[str, int]:
        """Render DB-selected entries that did not already render in the intercepted prompt."""
        state = self._current_static_selection_state()
        if state is None or not state.selected_ids:
            return "", 0
        by_parent: Dict[str, List[Any]] = {}
        parent_order: List[str] = []
        count = 0
        for sid in state.selected_order:
            if sid in state.rendered_ids:
                continue
            section = state.selected_sections_by_id.get(sid)
            if section is None:
                continue
            parent = str(getattr(section, "static_index_parent_title", "") or getattr(section, "selector_parent_title", "") or "Indexed GM").strip() or "Indexed GM"
            if parent not in by_parent:
                by_parent[parent] = []
                parent_order.append(parent)
            by_parent[parent].append(section)
            state.rendered_ids.add(sid)
            count += 1
        if not count:
            return "", 0
        rendered: List[str] = []
        for parent in parent_order:
            parent_header = self._render_parent_header(parent)
            if parent_header:
                rendered.append(parent_header)
            for section in by_parent[parent]:
                block = self._format_world_section(
                    str(getattr(section, "title", "") or parent),
                    str(getattr(section, "full_content", "") or ""),
                )
                if block:
                    rendered.append(block)
        return self._normalize_render_spacing("\n\n".join(rendered)), count

    def _prepare_all_static_gm_selector_candidates(
        self,
        sections: List[Any],
        request_type: str = "",
    ) -> tuple[List[Any], Dict[str, int], int, int]:
        """Prepare selector candidates from incoming GM sections plus DB-only GM entries.

        Static GM Index is the authoritative source for all configured [GM] child
        elements. Incoming prompt sections are still parsed because they carry pinned
        text and preserve in-place rendering for world.txt, but selector candidates must
        also include indexed entries from files AIInfluence did not inject into this
        prompt, such as actionrules.txt.
        """
        prepared: List[Any] = []
        parent_counts: Dict[str, int] = {}
        skipped_empty = 0
        added_from_db = 0
        parent_title = ""
        static_index = getattr(self.retriever, "static_gm_index", None)
        seen_static_ids: Set[str] = set()
        seen_prompt_keys: Set[tuple[str, str, str]] = set()

        def add_parent_count(title: str) -> None:
            group_title = title or "GM Block"
            parent_counts[group_title] = parent_counts.get(group_title, 0) + 1

        for section in sections:
            level = int(getattr(section, "level", 0) or 0)
            title = str(getattr(section, "title", "") or "").strip()

            if level == 1:
                parent_title = title
                if self.retriever._is_empty_container_section(section):
                    skipped_empty += 1
                    continue

            if self.retriever._is_empty_container_section(section):
                skipped_empty += 1
                continue

            indexed = None
            if static_index is not None and hasattr(static_index, "find_for_prompt_section"):
                try:
                    indexed = static_index.find_for_prompt_section(section, parent_title=parent_title)
                except Exception as exc:
                    logger.debug("Static GM index lookup failed for selector candidate %s/%s: %s", parent_title, title, exc)

            summary = ""
            if indexed is not None:
                summary = str(getattr(indexed, "summary", "") or "").strip()
                db_parent_title = str(getattr(indexed, "parent_title", "") or "").strip()
                static_id = str(getattr(indexed, "id", "") or "").strip()
                if static_id:
                    seen_static_ids.add(static_id)
                    setattr(section, "static_index_id", static_id)
                    setattr(section, "static_index_source", str(getattr(indexed, "source_file", "") or ""))
                    setattr(section, "static_index_summary_source", str(getattr(indexed, "summary_source", "") or ""))
                    setattr(section, "static_index_parent_title", db_parent_title)
            if not summary:
                summary = str(getattr(section, "selector_summary", "") or getattr(section, "summary", "") or "").strip()
            if not summary and static_index is not None and hasattr(static_index, "_deterministic_summary"):
                try:
                    summary = static_index._deterministic_summary(
                        title=title,
                        parent_title=parent_title,
                        full_content=str(getattr(section, "full_content", "") or ""),
                    )
                except Exception:
                    summary = ""
            if not summary:
                summary = self.retriever._generate_summary(section)

            setattr(section, "selector_summary", summary)
            setattr(section, "selector_parent_title", str(getattr(section, "static_index_parent_title", "") or "").strip() or parent_title)
            setattr(section, "selector_match_score", 1.0)
            setattr(section, "selector_match_reason", "all_gm_candidates_no_shortlist")
            setattr(section, "external_static_gm", False)

            content_hash = ""
            if static_index is not None and hasattr(static_index, "_hash_text") and hasattr(static_index, "_canonical_content"):
                try:
                    content_hash = static_index._hash_text(static_index._canonical_content(str(getattr(section, "full_content", "") or "")))
                except Exception:
                    content_hash = ""
            seen_prompt_keys.add((str(getattr(section, "static_index_source", "") or ""), parent_title.lower(), title.lower()))
            if content_hash:
                seen_prompt_keys.add(("hash", content_hash, ""))

            add_parent_count(parent_title or self._normalize_selector_block_title("", [section]) or title)
            prepared.append(section)

        # Add DB entries that were not present in the incoming prompt. This is the
        # key behavior for indexed files such as actionrules.txt: they can be selector
        # candidates even when AIInfluence did not inject them into ### The World ###.
        if static_index is not None and hasattr(static_index, "elements_for_request"):
            try:
                indexed_elements = static_index.elements_for_request(request_type)
            except Exception as exc:
                logger.warning("Static GM index candidate enumeration failed: %s", exc)
                indexed_elements = []

            for elem in indexed_elements:
                elem_id = str(getattr(elem, "id", "") or "").strip()
                if elem_id and elem_id in seen_static_ids:
                    continue
                parent = str(getattr(elem, "parent_title", "") or "").strip()
                child = str(getattr(elem, "child_title", "") or "").strip()
                source = str(getattr(elem, "source_file", "") or "").strip()
                content_hash = str(getattr(elem, "content_hash", "") or "").strip()
                if (source, parent.lower(), child.lower()) in seen_prompt_keys:
                    continue
                if content_hash and ("hash", content_hash, "") in seen_prompt_keys:
                    continue

                full_content = str(getattr(elem, "full_content", "") or "").strip()
                if not full_content:
                    continue

                summary = str(getattr(elem, "summary", "") or getattr(elem, "deterministic_summary", "") or "").strip()
                pseudo = Section(
                    title=child or parent or "Indexed GM Entry",
                    full_content=full_content,
                    summary=summary,
                    source=source or "static_gm_index",
                    level=2,
                    policy=self.retriever.POLICY_GM,
                    explicit_policy=self.retriever.POLICY_GM,
                )
                setattr(pseudo, "selector_summary", summary)
                setattr(pseudo, "selector_parent_title", parent or source or "Indexed GM")
                setattr(pseudo, "static_index_parent_title", parent)
                setattr(pseudo, "selector_match_score", 1.0)
                setattr(pseudo, "selector_match_reason", "db_static_gm_candidate")
                setattr(pseudo, "static_index_id", elem_id)
                setattr(pseudo, "static_index_source", source)
                setattr(pseudo, "static_index_summary_source", str(getattr(elem, "summary_source", "") or ""))
                setattr(pseudo, "external_static_gm", True)
                prepared.append(pseudo)
                if elem_id:
                    seen_static_ids.add(elem_id)
                if content_hash:
                    seen_prompt_keys.add(("hash", content_hash, ""))
                add_parent_count(parent or source or "Indexed GM")
                added_from_db += 1

        return prepared, parent_counts, skipped_empty, added_from_db

    def _render_external_static_gm_sections(self, sections: List[Any]) -> List[str]:
        """Render selected DB-only GM entries that were not present in the prompt."""
        if not sections:
            return []
        rendered: List[str] = []
        current_parent = None
        for section in sections:
            parent = str(getattr(section, "static_index_parent_title", "") or getattr(section, "selector_parent_title", "") or "Indexed GM").strip() or "Indexed GM"
            if parent != current_parent:
                parent_header = self._render_parent_header(parent)
                if parent_header:
                    rendered.append(parent_header)
                current_parent = parent
            block = self._format_world_section(str(getattr(section, "title", "") or parent), str(getattr(section, "full_content", "") or ""))
            if block:
                rendered.append(block)
        return rendered

    async def _select_policy_gm_sections_for_request(
        self,
        sections: List[Any],
        query: str,
        entities: Dict[str, List[str]],
        request_type: str,
        selector_context: Optional[List[tuple[str, str]]] = None,
        dialogue_query: str = "",
        block_title: str = "",
    ) -> List[Any]:
        """Select matching [GM] sections for every request type.

        The retriever parser has already converted non-matching scoped tags, such as
        [GM:EVENTS] during a dialogue request, into IGNORE. The remaining sections are eligible
        for semantic selection.
        """
        selection_query = self._build_gm_selection_query(
            fallback_query=query,
            dialogue_query=dialogue_query,
            context_segments=selector_context or [],
        )

        if self._static_gm_index_available():
            return await self._select_static_index_gm_sections_for_request(
                sections=list(sections),
                query=selection_query or query,
                entities=entities,
                request_type=request_type,
                selector_context=selector_context or [],
                dialogue_query=dialogue_query,
                block_title=block_title,
            )

        candidate_sections = list(sections)
        if self._selector_enabled():
            selected = await self._select_via_selector_model(
                sections=candidate_sections,
                request_type=request_type,
                dialogue_query=dialogue_query,
                context_segments=selector_context or [],
                block_title=block_title,
                fallback_query=selection_query or query,
                fallback_entities=entities,
            )
            if self.selector_client:
                selected_parent_counts: Dict[str, int] = {}
                selected_external = 0
                for sec in selected:
                    parent = str(getattr(sec, "selector_parent_title", "") or "GM Block").strip() or "GM Block"
                    selected_parent_counts[parent] = selected_parent_counts.get(parent, 0) + 1
                    if bool(getattr(sec, "external_static_gm", False)):
                        selected_external += 1
                await self.selector_client.log_diagnostic(
                    "STATIC GM SELECTED FOR MAIN PROMPT",
                    {
                        "request_type": request_type,
                        "selector_candidates": len(candidate_sections),
                        "selected_total": len(selected),
                        "selected_db_only": selected_external,
                        "selected_prompt_present": len(selected) - selected_external,
                        "selected_parent_counts": selected_parent_counts,
                    },
                )
            return selected
        # Fail closed when no selector backend is available. Returning all GM sections here
        # is worse than omitting optional GM context because it can send entire indexed files
        # to the main model. Pinned content is still handled separately by the render path.
        logger.error("No selector backend is enabled during [GM] selection; fail-closed with zero GM selections")
        if self.selector_client:
            try:
                await self.selector_client.log_diagnostic(
                    "SELECTOR UNAVAILABLE FAIL-CLOSED",
                    {
                        "request_type": request_type,
                        "candidate_count": len(candidate_sections),
                        "reason": "selector backend disabled or unavailable",
                    },
                )
            except Exception:
                pass
        return []

    def _normalize_render_spacing(self, text: str) -> str:
        """Collapse excessive blank lines introduced by section joins while keeping section breaks."""
        cleaned = text.strip()
        if not cleaned:
            return ''
        return re.sub(r'(?:\r?\n){3,}', '\n\n', cleaned)

    def _repair_split_policy_headers(self, text: str) -> str:
        """Repair malformed policy headers that were split across physical lines.

        Example:
            ===\n\n [GM:DIALOGUE] Forms of Service ===
        becomes:
            === [GM:DIALOGUE] Forms of Service ===
        """
        if not text:
            return ''
        repaired = re.sub(
            r'(={3,})[ \t]*(?:\r?\n)+[ \t]*(\[(?:PINNED|PIN|GM|IGNORE|END\s+(?:PINNED|PIN|GM|IGNORE))(?::[A-Z0-9_ -]+)*\][ \t]*.*?[ \t]*={3,})',
            r'\1 \2',
            text,
            flags=re.IGNORECASE,
        )
        return repaired

    def _normalize_policy_header_seams(self, text: str) -> str:
        """Ensure surviving policy headers are separated from neighboring prose."""
        if not text:
            return ''
        normalized = re.sub(
            r'(?<!\n)(?=(?:={3,}[ \t]*\[(?:PINNED|PIN|GM|IGNORE|END\s+(?:PINNED|PIN|GM|IGNORE))(?::[A-Z0-9_ -]+)*\][ \t]*.*?[ \t]*={3,}))',
            '\n\n',
            text,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r'(={3,}[ \t]*\[(?:PINNED|PIN|GM|IGNORE|END\s+(?:PINNED|PIN|GM|IGNORE))(?::[A-Z0-9_ -]+)*\][ \t]*.*?[ \t]*={3,})(?=[^\r\n])',
            r'\1\n\n',
            normalized,
            flags=re.IGNORECASE,
        )
        return self._normalize_render_spacing(normalized)

    def _finalize_visible_policy_headers(self, text: str) -> str:
        """Final scrub: remove residual policy markers only.

        Older versions tried to infer/drop orphaned headers here.  That broke compact
        header formats because parent titles could disappear when the next line was a
        child ``##`` header.  Inclusion/removal decisions have already happened before
        this point, so finalization must only remove control markers.
        """
        return self._clean_visible_policy_text(text)

    def _text_has_policy_header(self, text: str) -> bool:
        return bool(re.search(
            r'(?im)^\s*(?:={3,}\s*\[(?:PINNED|PIN|GM|IGNORE)(?::[A-Z0-9_ -]+)*\]\s*.*?={3,}|\[(?:PINNED|PIN|GM|IGNORE)(?::[A-Z0-9_ -]+)*\]\s*.*?)\s*$',
            text or "",
        ))

    async def _filter_policy_regions_inside_outer_section(
        self,
        raw: str,
        selection_query: str,
        entities: Dict[str, List[str]],
        request_type: str,
        selector_context: List[tuple[str, str]],
        dialogue_query: str,
    ) -> tuple[str, int, int]:
        """Filter [PINNED]/[GM]/[IGNORE] regions inside an outer AIInfluence section.

        Unknown AIInfluence sections used to be returned raw. If such a section contained
        a configured text file with [GM] headers, every unselected child leaked to the main
        model. This keeps the outer section in place but filters any policy-tagged regions
        inside it.
        """
        if not raw.strip() or not self._text_has_policy_header(raw):
            return raw.strip(), 0, 0

        lines = raw.splitlines()
        out: List[str] = []
        i = 0
        selected_total = 0
        pinned_total = 0

        while i < len(lines):
            line = lines[i]
            if self._is_policy_header_line(line):
                start = i
                i += 1
                if not self._line_has_inline_policy_end(line):
                    while i < len(lines):
                        if self._is_policy_end_marker_line(lines[i]):
                            i += 1
                            break
                        if self._is_policy_header_line(lines[i]):
                            break
                        i += 1
                raw_policy = '\n'.join(lines[start:i]).strip()
                filtered, selected, pinned = await self._filter_tagged_policy_block_preserve_order(
                    raw_policy,
                    selection_query=selection_query,
                    entities=entities,
                    request_type=request_type,
                    selector_context=selector_context,
                    dialogue_query=dialogue_query,
                )
                if filtered:
                    if out and out[-1].strip():
                        out.append('')
                    out.extend(filtered.splitlines())
                selected_total += selected
                pinned_total += pinned
                continue
            if self._is_policy_end_marker_line(line):
                i += 1
                continue
            out.append(line)
            i += 1

        return self._normalize_render_spacing('\n'.join(out)), selected_total, pinned_total

    async def _replace_known_prompt_spans_in_place(
        self,
        text: str,
        focus_text: str,
        selection_query: str,
        entities: Dict[str, List[str]],
        request_type: str,
        context_content: str,
        selector_context: List[tuple[str, str]],
        dialogue_query: str,
    ) -> tuple[str, int, int]:
        """
        Replace only known filterable prompt spans while preserving all other text and order.

        The scanner treats AIInfluence ### headings as replaceable prompt spans. It has a
        special case for ### The World ### so internal world.txt === / ## headings do not
        cause the block to be reordered or split by proxy-owned group names.
        """
        lines = text.splitlines()
        out: List[str] = []
        i = 0
        sections_included = 0
        pinned_count = 0

        def append_chunk(chunk: str) -> None:
            if not chunk:
                return
            if out and out[-1] != '':
                out.append('')
            out.append(chunk)

        while i < len(lines):
            line = lines[i]
            title = self._normalize_section_title(line)

            # Explicit tagged blocks outside ### The World ###, such as actionrules/event rules
            # inserted into larger prompts. These are filtered in place and keep their original
            # location. A top-level === [GM] block includes its ## children until the next
            # top-level AIInfluence/header boundary.
            if self._is_policy_header_line(line):
                start = i
                i += 1
                if not self._line_has_inline_policy_end(line):
                    while i < len(lines):
                        if self._is_policy_end_marker_line(lines[i]):
                            i += 1
                            break
                        if self._is_policy_header_line(lines[i]) or self._is_aiinfluence_section_header(lines[i]):
                            break
                        i += 1
                raw = '\n'.join(lines[start:i]).strip()
                filtered_policy, selected, pinned = await self._filter_tagged_policy_block_preserve_order(
                    raw,
                    selection_query=selection_query,
                    entities=entities,
                    request_type=request_type,
                    selector_context=selector_context,
                    dialogue_query=dialogue_query,
                )
                if filtered_policy:
                    append_chunk(filtered_policy)
                sections_included += selected
                pinned_count += pinned
                continue

            if title == 'the world':
                start = i
                i += 1
                while i < len(lines):
                    next_title = self._normalize_section_title(lines[i])
                    # Stable AIInfluence sections that follow the world block. Internal world.txt
                    # headers such as === [PINNED] ... === and nested ## entries stay inside it.
                    if next_title in {'global politics of the world', 'character briefing current data'}:
                        break
                    i += 1
                raw = '\n'.join(lines[start:i]).strip()
                filtered_world, selected, pinned = await self._filter_single_incoming_world_block_preserve_order(
                    raw,
                    selection_query=selection_query,
                    entities=entities,
                    request_type=request_type,
                    selector_context=selector_context,
                    dialogue_query=dialogue_query,
                )
                if filtered_world:
                    append_chunk(filtered_world)
                sections_included += selected
                pinned_count += pinned
                continue

            if self._is_aiinfluence_section_header(line):
                start = i
                i += 1
                while i < len(lines) and not self._is_aiinfluence_section_header(lines[i]):
                    i += 1
                raw = '\n'.join(lines[start:i]).strip()
                if self._text_has_policy_header(raw):
                    replacement, selected, pinned = await self._filter_policy_regions_inside_outer_section(
                        raw,
                        selection_query=selection_query,
                        entities=entities,
                        request_type=request_type,
                        selector_context=selector_context,
                        dialogue_query=dialogue_query,
                    )
                    sections_included += selected
                    pinned_count += pinned
                else:
                    replacement = self._filter_known_ai_section(raw, focus_text, request_type, context_content)
                if replacement:
                    append_chunk(replacement)
                continue

            # Closing markers are parser controls only; remove them if they remain in free text.
            if self._is_policy_end_marker_line(line):
                i += 1
                continue

            out.append(line)
            i += 1

        return '\n'.join(out), sections_included, pinned_count

    def _is_aiinfluence_section_header(self, line: str) -> bool:
        """True for AIInfluence-style prompt-section headers. Single # is not a filter boundary."""
        return bool(
            re.match(r'^\s*#{3,6}\s+.+?\s*$', line)
            or re.match(r'^\s*={3,}\s*[^=\s].*?\s*={3,}\s*$', line)
        )

    def _filter_known_ai_section(self, raw: str, focus_text: str, request_type: str, context_content: str) -> str:
        """Filter one known AIInfluence ### section; unknown sections are returned unchanged."""
        if not raw.strip():
            return ''
        header = raw.splitlines()[0]
        title = self._normalize_section_title(header)

        if title == 'global politics of the world':
            return raw.strip()
        if title == 'character briefing current data':
            return self._filter_character_briefing(raw, focus_text)
        if title == 'immediate situation current data':
            return raw.strip()
        if title == 'the player current data':
            return self._filter_player_data(raw, focus_text)
        if title == 'people physically present in this location right now':
            return self._filter_people_present(raw, focus_text)
        if title == 'nearby settlements strategic context current data':
            return self._filter_nearby_list(raw, focus_text, kind='settlement')
        if title == 'nearby parties npc vicinity current data':
            return self._filter_nearby_list(raw, focus_text, kind='party')
        if title == 'mentioned settlements':
            return self._filter_nearby_list(raw, focus_text, kind='settlement')
        if title == 'mentioned characters':
            return self._filter_mentioned_records(raw, focus_text, label='character')
        if title == 'mentioned parties':
            return self._filter_nearby_list(raw, focus_text, kind='party')
        if title == 'conversation history':
            return self._filter_plain_conversation_history(raw)

        # Event/diplomacy prompt sections.
        if title in {'current world data ground truth', 'current world state'}:
            return raw.strip()
        if title in {'existing events historical narrative', 'existing dynamic events do not duplicate'}:
            return self._filter_bulleted_entries(raw, max_entries=self.MAX_EVENT_HISTORY)
        if title == 'active economic effects':
            return self._filter_bulleted_entries(raw, max_entries=self.max_nearby_settlements)
        if title == 'recent diplomatic statements last 15 statements from last 50 days':
            return self._filter_recent_diplomatic_statements(raw)
        if title == 'new npc dialogues since last analysis':
            return self._filter_new_npc_dialogues(raw, context_content=context_content)

        # Unknown sections stay exactly where AIInfluence put them.
        return raw.strip()

    async def _filter_single_incoming_world_block_preserve_order(
        self,
        block: str,
        selection_query: str,
        entities: Dict[str, List[str]],
        request_type: str = "dialogue",
        selector_context: Optional[List[tuple[str, str]]] = None,
        dialogue_query: str = "",
    ) -> tuple[str, int, int]:
        """Filter one incoming ### The World ### block without reordering its sections."""
        lines = block.splitlines()
        header = lines[0] if lines else '### The World ###'
        body = '\n'.join(lines[1:]).strip()
        if not body:
            return header, 0, 0

        body = self._repair_split_policy_headers(body)
        body = self._normalize_policy_block_boundaries(body)

        sections = self.retriever._parse_sections(body, 'incoming_world', request_type=request_type)
        promoted_titles = self._promote_inferred_structured_top_level_world_sections(sections)
        gm_sections = []
        pinned_count = 0
        for sec in sections:
            if sec.policy == self.retriever.POLICY_IGNORE:
                continue
            is_empty = self.retriever._is_empty_container_section(sec)
            if is_empty and not (self._selection_backend_enabled() and sec.policy == self.retriever.POLICY_GM and sec.level == 1):
                continue
            sec.entities = self.retriever._extract_entities(sec.full_content)
            sec.summary = self.retriever._generate_summary(sec)
            if sec.policy == self.retriever.POLICY_PINNED:
                pinned_count += 1
            else:
                gm_sections.append(sec)

        if self._selector_enabled() and self.selector_client:
            await self.selector_client.log_diagnostic(
                "SELECTOR CANDIDATE DISCOVERY",
                self._build_selector_candidate_discovery_summary(
                    block_header=header,
                    request_type=request_type,
                    sections=sections,
                    promoted_titles=promoted_titles,
                ),
            )

        selected = await self._select_policy_gm_sections_for_request(
            gm_sections,
            selection_query,
            entities,
            request_type,
            selector_context=selector_context or [],
            dialogue_query=dialogue_query,
            block_title=header,
        )
        selected_ids = {id(sec) for sec in selected}
        body_rendered, selected_count = self._render_selected_world_sections(sections, selected_ids)
        rendered = [header] + body_rendered
        return '\n\n'.join(rendered).strip(), selected_count, pinned_count

    def _promote_inferred_structured_top_level_world_sections(self, sections: List[Any]) -> List[str]:
        """Recover selector candidates when an upstream prompt lost a top-level [GM:...] tag.

        AIInfluence sometimes appears to strip a policy marker from a major world header while
        leaving its `##` children intact. In that case the parser sees an untagged level-1
        section, which normally defaults to pinned and never reaches the selector. When the
        section is clearly just a header container for child entries, promote that top-level
        block and its inherited children back to GM so selector filtering can still happen.
        """
        if not sections or not self._selection_backend_enabled():
            return []

        direct_children: Dict[int, List[Any]] = {}
        current_parent = None
        for sec in sections:
            if getattr(sec, "policy", "") == self.retriever.POLICY_IGNORE:
                continue
            if int(getattr(sec, "level", 0) or 0) == 1:
                current_parent = sec
                direct_children[id(sec)] = []
                continue
            if current_parent is not None:
                direct_children.setdefault(id(current_parent), []).append(sec)

        promoted_titles: List[str] = []
        for sec in sections:
            if int(getattr(sec, "level", 0) or 0) != 1:
                continue
            if getattr(sec, "policy", "") != self.retriever.POLICY_PINNED:
                continue
            if getattr(sec, "explicit_policy", None) is not None:
                continue
            if not self.retriever._is_empty_container_section(sec):
                continue

            children = direct_children.get(id(sec), [])
            if not children:
                continue
            if not any(int(getattr(child, "level", 0) or 0) > 1 for child in children):
                continue

            sec.policy = self.retriever.POLICY_GM
            promoted_titles.append(str(getattr(sec, "title", "") or "(untitled)").strip())
            for child in children:
                if getattr(child, "policy", "") != self.retriever.POLICY_PINNED:
                    continue
                if getattr(child, "explicit_policy", None) is not None:
                    continue
                child.policy = self.retriever.POLICY_GM

        return promoted_titles

    def _build_selector_candidate_discovery_summary(
        self,
        *,
        block_header: str,
        request_type: str,
        sections: List[Any],
        promoted_titles: List[str],
    ) -> Dict[str, Any]:
        """Summarize parsed top-level world blocks before selector batching."""
        top_levels: List[Dict[str, Any]] = []
        current_entry: Optional[Dict[str, Any]] = None
        for sec in sections:
            if getattr(sec, "policy", "") == self.retriever.POLICY_IGNORE:
                continue
            level = int(getattr(sec, "level", 0) or 0)
            if level == 1:
                current_entry = {
                    "title": str(getattr(sec, "title", "") or ""),
                    "policy": str(getattr(sec, "policy", "") or ""),
                    "explicit_policy": getattr(sec, "explicit_policy", None),
                    "is_empty_container": bool(self.retriever._is_empty_container_section(sec)),
                    "child_count": 0,
                }
                top_levels.append(current_entry)
                continue
            if current_entry is not None:
                current_entry["child_count"] += 1

        return {
            "block_header": block_header,
            "request_type": request_type,
            "promoted_to_gm": promoted_titles,
            "top_level_sections": top_levels,
        }

    def _assemble_filtered_content(
        self,
        retrieved: Dict[str, Any],
        retrieved_world_sections: List[SelectedSection],
        world_section_count: int,
        game_state: Optional[Dict[str, Any]],
        event_history: Optional[List[Dict[str, Any]]],
        dialogue_history: Optional[List[Any]],
        ai_sections: Dict[str, str],
        entities: Dict[str, List[str]],
        incoming_world_lore: str = '',
    ) -> Dict[str, str]:
        """Assemble filtered sections. All values are complete sections/objects, not string slices."""
        filtered_content: Dict[str, str] = {}

        # 0. World lore. In v9, world/rules/traditions are NEVER loaded from disk.
        # If AIInfluence injected ### The World ###, filter that incoming block in place.
        # If it did not, do not add any external fallback content.
        if incoming_world_lore:
            filtered_content['world_lore'] = incoming_world_lore

        # 1. Game state from the intercepted prompt only
        if game_state:
            filtered_game_state = self._filter_game_state(game_state, entities)
            filtered_content['game_state'] = json.dumps(filtered_game_state, indent=2, ensure_ascii=False)

        # 5. Event history from system prompt only
        if event_history:
            filtered_history = self._filter_event_history(event_history, entities)
            filtered_content['event_history'] = json.dumps(filtered_history, indent=2, ensure_ascii=False)

        # 6. Dialogue/conversation history from system prompt: preserve latest N whole entries.
        if dialogue_history and self.DIALOGUE_HISTORY_SIZE > 0:
            preserved_dialogue = dialogue_history[-self.DIALOGUE_HISTORY_SIZE:]
            filtered_content['dialogue_history'] = json.dumps(preserved_dialogue, indent=2, ensure_ascii=False)

        # 7. AIInfluence hardcoded prompt sections. The mod cannot be edited, so the proxy
        # parses its emitted markdown-ish sections into complete pseudo-sections.
        for key, value in ai_sections.items():
            if value:
                filtered_content[key] = value

        return filtered_content

    def _build_selection_query(
        self,
        context_content: str,
        dialogue_query: str,
        source_content: str,
        request_type: str,
        game_state: Optional[Dict[str, Any]],
        event_history: Optional[List[Dict[str, Any]]],
        dialogue_history: Optional[List[Any]],
        ai_relevance_signal: str = '',
    ) -> str:
        """Build a compact semantic query without dumping the whole prompt-container into selection."""
        parts: List[str] = []
        if request_type == "dialogue":
            if dialogue_query.strip():
                # Dialogue selection should key off only the player's actual last spoken line.
                parts.append(dialogue_query.strip())
        else:
            # Events/diplomacy: the prompt-container is the only user message. Use only a compact
            # task/focus hint, never the full world/game-data blob.
            focus_hint = self._extract_request_focus(source_content, request_type)
            if focus_hint:
                parts.append(focus_hint)
        # Do not add request_type by itself; generic words like 'dialogue' make unrelated
        # world sections look weakly relevant. Structured state/dialogue below is enough when present.

        # Do not add extracted world-state/history into the world-lore selector query. Those
        # sections contain many kingdoms/items/IDs and make unrelated GM sections look relevant.
        return self._join_nonempty(parts)

    def _collapse_duplicate_visible_wrappers(self, text: str) -> str:
        """Repair doubled wrappers from older render paths without changing authored style."""
        value = str(text or "").strip()
        for _ in range(3):
            previous = value
            value = re.sub(r'^(={2,})\s*((?:={2,})\s*.+?\s*(?:={2,}))\s*\1$', r'\2', value).strip()
            value = re.sub(r'^(#{1,6})\s+((?:#{1,6})\s+.+?)$', r'\2', value).strip()
            if value == previous:
                break
        return value

    def _clean_visible_policy_text(self, text: str) -> str:
        """Remove policy-control markers from text already chosen to render.

        This function does not decide what to include and does not invent headings.
        It only strips control markers such as [GM:DIALOGUE], [PINNED], [IGNORE],
        and [END ...], preserving the author's remaining title/body formatting.
        """
        if not text:
            return ""

        end_marker_line_re = re.compile(
            r'^\s*(?:={2,}\s*)?\[END\s+(?:PINNED|PIN|GM|IGNORE)(?::[A-Z0-9_ -]+)*\](?:\s*={2,})?\s*\.?\s*$',
            re.IGNORECASE,
        )
        open_marker_re = re.compile(
            r'\[(?:PINNED|PIN|GM|IGNORE)(?::[A-Z0-9_ -]+)*\]\s*',
            re.IGNORECASE,
        )
        inline_end_re = re.compile(
            r'\s*\[END\s+(?:PINNED|PIN|GM|IGNORE)(?::[A-Z0-9_ -]+)*\]\s*\.?',
            re.IGNORECASE,
        )

        cleaned_lines: List[str] = []
        for raw in str(text or "").splitlines():
            line = raw.rstrip()
            if end_marker_line_re.match(line):
                continue
            line = open_marker_re.sub('', line)
            line = inline_end_re.sub('', line)
            line = self._collapse_duplicate_visible_wrappers(line.rstrip())
            cleaned_lines.append(line)

        return self._normalize_render_spacing('\n'.join(cleaned_lines))

    def _meaningful_parent_header(self, header: str) -> bool:
        """Return True only when a rendered parent header contains real title text."""
        cleaned = self._clean_visible_policy_text(str(header or "")).strip()
        if not cleaned:
            return False
        # Reject pure decoration/separators such as "#", "---", "====", or "...".
        return bool(re.search(r'[A-Za-z0-9]{2,}', cleaned))

    def _db_parent_title_for_section(self, section: Any) -> str:
        """Best DB-authored parent title known for a selected child section."""
        return str(
            getattr(section, "static_index_parent_title", "")
            or getattr(section, "selector_parent_title", "")
            or ""
        ).strip()

    def _render_parent_header(self, title: str) -> str:
        """Render a selected DB parent title exactly as authored/stored, without wrapping."""
        return self._clean_visible_policy_text(str(title or "").strip())

    def _render_header_only_for_section(self, section: Any) -> str:
        """Return only the authored visible header line for a parsed parent section."""
        first_line = ""
        for raw in str(getattr(section, "full_content", "") or "").splitlines():
            stripped = raw.strip()
            if stripped:
                first_line = stripped
                break
        if first_line:
            return self._clean_visible_policy_text(first_line)
        return self._render_parent_header(str(getattr(section, "title", "") or ""))

    def _parent_header_for_child(self, section: Any, current_parent: Any = None) -> str:
        """Resolve the visible parent header that should precede a selected child.

        Prefer the parsed parent header when available so compact authored formats such as
        ``[GM:DIALOGUE] ### Religions ###`` render as ``### Religions ###``.  Fall back
        to the static-index parent title stored on the selected child; this covers DB-only
        candidates and prompt blocks where the empty parent container was not present in
        the local render walk.
        """
        db_parent = self._db_parent_title_for_section(section)
        db_header = self._render_parent_header(db_parent)

        if current_parent is not None:
            header = self._render_header_only_for_section(current_parent)
            # Use the live authored parent only when it is a real title.  If AIInfluence or
            # a malformed separator left us with a parent like "#", fall back to the DB
            # parent title indexed from the configured text file.
            if self._meaningful_parent_header(header):
                return header

        if self._meaningful_parent_header(db_header):
            return db_header
        return db_header or self._render_parent_header(str(getattr(section, "selector_parent_title", "") or ""))

    def _parent_render_key(self, header: str) -> str:
        return re.sub(r'\s+', ' ', self._clean_visible_policy_text(header or '')).strip().lower()

    def _format_world_section(self, title: str, full_content: str) -> str:
        """Format an already-selected section without inventing any app headings."""
        return self._clean_visible_policy_text(str(full_content or "").lstrip())

    def _normalize_policy_block_boundaries(self, text: str) -> str:
        """Ensure authored policy/child headers are physically separable before parsing.

        This is intentionally block-local, not global prompt rewriting. It fixes cases where
        a boundary was accidentally flattened into:
            "... break.=== [GM:DIALOGUE] Forms of Service ==="
        or
            "... legends## Braavosi"
        which would otherwise cause the parser to swallow later GM children into the
        previous section/policy.
        """
        if not text:
            return ''

        normalized = text.replace('\r\n', '\n')

        # Policy headers / END markers glued to preceding content.
        normalized = re.sub(
            r'(?<!\n)(?=(?:={3,}[ \t]*\[(?:PINNED|PIN|GM|IGNORE|END\s+(?:PINNED|PIN|GM|IGNORE))(?::[A-Z0-9_ -]+)*\][ \t]*.*?[ \t]*={3,}))',
            '\n\n',
            normalized,
            flags=re.IGNORECASE,
        )
        # Child markdown headers glued to preceding content.
        #
        # Important: do not split inside authored markdown headings such as
        # ``### Summary ###``.  The old pattern looked for ``##`` anywhere not
        # preceded by a newline, so it rewrote ``### Summary ###`` into
        # ``#\n\n## Summary\n\n#\n\n##``.  Only insert a boundary when the
        # hash run is glued to normal prose, e.g. ``... legends## Braavosi``.
        normalized = re.sub(r'(?<![\n#\s\]])(?=(?:#{2,6}\s+))', '\n\n', normalized)
        # Header lines glued to following content.
        normalized = re.sub(
            r'(={3,}[ \t]*\[(?:PINNED|PIN|GM|IGNORE|END\s+(?:PINNED|PIN|GM|IGNORE))(?::[A-Z0-9_ -]+)*\][ \t]*.*?[ \t]*={3,})(?=[^\n])',
            r'\1\n\n',
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(r'(?:\n){3,}', '\n\n', normalized)
        return normalized.strip()

    def _count_world_sections(self, filtered_content: Dict[str, str]) -> int:
        world = filtered_content.get('world_lore')
        if not world:
            return 0
        return len(re.findall(r'(?m)^\s*(?:={2,}\s*.+?\s*={2,}|#{1,6}\s*.+?)\s*$', world)) or 1

    def _extract_target_content(self, messages: List[Dict[str, Any]], target_message_index: Optional[int]) -> str:
        """Extract the selected prompt-container message content."""
        if target_message_index is not None and 0 <= target_message_index < len(messages):
            return self._content_to_text(messages[target_message_index].get('content', ''))
        return self._extract_content(messages, roles={'system'})

    def _extract_context_content(self, messages: List[Dict[str, Any]], target_message_index: Optional[int]) -> str:
        """Extract text from messages other than the selected prompt-container."""
        parts: List[str] = []
        for i, msg in enumerate(messages):
            if target_message_index is not None and i == target_message_index:
                continue
            text = self._content_to_text(msg.get('content', ''))
            if text:
                parts.append(text)
        return '\n'.join(parts)

    def _extract_last_user_message(self, messages: List[Dict[str, Any]]) -> str:
        """Return the player's actual last spoken line from the last user message."""
        for msg in reversed(messages):
            if msg.get('role') != 'user':
                continue
            text = self._content_to_text(msg.get('content', ''))
            if not text.strip():
                continue

            cleaned = text.strip()
            cleaned = re.split(
                r'\n\s*\n\s*Respond to this message in character\b',
                cleaned,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
            lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
            if not lines:
                continue
            for line in reversed(lines):
                if line.lower().startswith('player:'):
                    spoken = line.split(':', 1)[1].strip()
                    if spoken:
                        return spoken
            return cleaned
        return ''

    def _merge_entities(self, base: Dict[str, List[str]], extra: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Merge entity maps while preserving order and avoiding duplicates."""
        merged: Dict[str, List[str]] = {
            key: list(values)
            for key, values in base.items()
        }
        for key, values in extra.items():
            bucket = merged.setdefault(key, [])
            seen = {str(item).lower() for item in bucket if item}
            for value in values:
                if not value:
                    continue
                norm = str(value).lower()
                if norm in seen:
                    continue
                seen.add(norm)
                bucket.append(value)
        return merged

    def _extract_request_focus(self, source_content: str, request_type: str) -> str:
        """
        Extract a compact relevance hint from a user-only event/diplomacy prompt.

        This is only used for selector-side matching; it is never sent to the backend model by itself.
        It deliberately prefers the tail and explicit task-like lines because AIInfluence usually puts
        the actual event/diplomacy instruction near the end of the large prompt.
        """
        if not source_content.strip():
            return ''

        dialogue_focus = self._extract_event_dialogue_focus(source_content)
        if dialogue_focus:
            return dialogue_focus

        task_lines: List[str] = []
        task_re = re.compile(
            r'\b(generate|create|write|analy[sz]e|decide|choose|event|diplomacy|statement|response|action|quest|dialogue|scene|war|peace|alliance|kingdom)\b',
            re.IGNORECASE,
        )
        for line in source_content.splitlines():
            stripped = line.strip()
            if stripped and task_re.search(stripped):
                task_lines.append(stripped)

        # Prefer explicit task-like lines near the end. If none exist, use the tail as a selector hint.
        if task_lines:
            return '\n'.join(task_lines[-30:])

        tail_lines = [line.strip() for line in source_content.splitlines() if line.strip()]
        return '\n'.join(tail_lines[-30:])

    def _extract_event_dialogue_focus(self, source_content: str) -> str:
        """Prefer concrete dialogue/event material over boilerplate for event/diplomacy selector matching.

        Event prompts are one giant user message. The useful relevance signal is usually the
        latest public NPC dialogue block, not the large instructions/world/current-kingdom lists.
        """
        if not source_content:
            return ''
        lower = source_content.lower()
        starts = []
        for needle in ('new npc dialogues', 'new dialogue with ', 'mode: dialogue analysis'):
            idx = lower.rfind(needle)
            if idx != -1:
                starts.append(idx)
        if not starts:
            return ''
        start = max(starts)
        # Stop before the next major technical section if one appears after the selected focus.
        stop_candidates = []
        for pat in (r'\n={3,}\s*[A-Z][A-Z _()\-]+\s*={3,}', r'\n##\s*===\s*CURRENT WORLD DATA', r'\n###\s+OVERRIDE RULES'):
            m = re.search(pat, source_content[start + 1:], re.IGNORECASE)
            if m:
                stop_candidates.append(start + 1 + m.start())
        end = min(stop_candidates) if stop_candidates else min(len(source_content), start + 16000)
        lines = [ln.strip() for ln in source_content[start:end].splitlines() if ln.strip()]
        # Keep tail and header-ish lines so the query is compact but still contains names/places/actions.
        if len(lines) > 120:
            lines = lines[:20] + lines[-100:]
        return '\n'.join(lines)

    def _extract_content(self, messages: List[Dict[str, Any]], roles: Optional[Set[str]] = None) -> str:
        """Extract text content from selected message roles."""
        parts = []
        for msg in messages:
            role = msg.get('role', '')
            if roles is not None and role not in roles:
                continue
            text = self._content_to_text(msg.get('content', ''))
            if text:
                parts.append(text)
        return '\n'.join(parts)

    def _content_to_text(self, content: Any) -> str:
        """Convert OpenAI-style message content to plain text."""
        if content is None:
            return ''
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        parts.append(str(item.get('text', '')))
                    elif 'text' in item:
                        parts.append(str(item.get('text', '')))
                elif isinstance(item, str):
                    parts.append(item)
            return '\n'.join(p for p in parts if p)
        return str(content)

    def _join_nonempty(self, parts: Iterable[str]) -> str:
        return '\n'.join(part for part in parts if part)

    def _selector_enabled(self) -> bool:
        return bool(self.selector_client and self.selector_client.enabled)

    def _static_gm_index_available(self) -> bool:
        return bool(
            self.static_gm_index_enabled
            and self.retriever
            and getattr(self.retriever, "static_gm_index", None)
        )

    def _selection_backend_enabled(self) -> bool:
        return self._selector_enabled() or self._static_gm_index_available()

    def _build_gm_selection_query(
        self,
        *,
        fallback_query: str,
        dialogue_query: str,
        context_segments: List[tuple[str, str]],
    ) -> str:
        """Build a compact semantic query for static GM selection."""
        parts: List[str] = []
        base = dialogue_query.strip() or fallback_query.strip()
        if base:
            parts.append(base)

        for label, text in context_segments:
            if not str(text or "").strip():
                continue
            relevance_lines = self._extract_relevance_lines(text)
            if relevance_lines:
                parts.append(f"{label}: " + " | ".join(relevance_lines[:12]))
            else:
                compact = re.sub(r"\s+", " ", text).strip()
                if compact:
                    parts.append(f"{label}: {compact[:500]}")

        return self._join_nonempty(parts)

    def _extract_selector_span(
        self,
        text: str,
        beginning: str,
        end: str,
        include_beginning_marker: bool = True,
        include_end_marker: bool = False,
    ) -> str:
        """Extract a marker-bounded span for selector context."""
        if not text or not beginning or not end:
            return ""
        start = text.find(beginning)
        if start == -1:
            return ""
        end_index = text.find(end, start + len(beginning))
        if end_index == -1:
            return ""

        slice_start = start if include_beginning_marker else start + len(beginning)
        slice_end = end_index + len(end) if include_end_marker else end_index
        if slice_end <= slice_start:
            return ""
        return text[slice_start:slice_end].strip()

    def _strip_policy_regions_from_selector_context(self, text: str) -> str:
        """Remove static policy-tagged regions from text sent as selector context.

        The selector should judge indexed-entry summaries against live request context.
        It should not receive static [PINNED], [GM], or [IGNORE] blocks through a
        broad selector-context rule. Those blocks are handled separately.
        """
        if not text:
            return ""
        policy_header_re = re.compile(
            r'^\s*(?:={3,}\s*\[(?:PINNED|PIN|GM|IGNORE)(?::[^\]]+)*\]\s*.*?={3,}|\[(?:PINNED|PIN|GM|IGNORE)(?::[^\]]+)*\]\s*.*?)\s*$',
            re.IGNORECASE,
        )
        policy_end_re = re.compile(
            r'^\s*(?:={3,}\s*)?\[END\s+(?:PINNED|PIN|GM|IGNORE)(?::[^\]]+)*\](?:\s*={3,})?\s*\.?\s*$',
            re.IGNORECASE,
        )
        lines = str(text).splitlines()
        cleaned: List[str] = []
        skipping = False
        for raw in lines:
            stripped = raw.strip()
            if policy_header_re.match(stripped):
                # One-line compact policy entries are still static authored content; do not send
                # them to selector as live context. The line is consumed completely.
                skipping = not bool(re.search(r'\[END\s+(?:PINNED|PIN|GM|IGNORE)', stripped, re.IGNORECASE))
                continue
            if skipping:
                if policy_end_re.match(stripped):
                    skipping = False
                    continue
                if re.match(r'^\s*###\s+.+?\s+###\s*$', raw):
                    skipping = False
                    cleaned.append(raw)
                    continue
                continue
            cleaned.append(raw)

        result = "\n".join(cleaned).strip()
        result = re.sub(
            r'={3,}\s*\[(?:PINNED|PIN|GM|IGNORE)(?::[^\]]+)*\]\s*.*?={3,}.*?(?=(?:={3,}\s*\[(?:PINNED|PIN|GM|IGNORE|END\s+(?:PINNED|PIN|GM|IGNORE))|\n\s*###|\Z))',
            '',
            result,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return re.sub(r'\n{3,}', '\n\n', result).strip()

    def _clip_selector_context(self, text: str, limit: int = 3000) -> str:
        """Keep selector context bounded so candidate summaries dominate the request."""
        cleaned = re.sub(r'\s+', ' ', str(text or '')).strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: max(0, limit - 3)].rstrip() + '...'

    def _clip_selector_summary(self, text: str, limit: int = 700) -> str:
        """Hard cap per-candidate summary text sent to the selector."""
        cleaned = re.sub(r'\s+', ' ', str(text or '')).strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: max(0, limit - 3)].rstrip() + '...'

    def _build_selector_context_segments(
        self,
        source_content: str,
        context_content: str,
        dialogue_query: str,
        request_type: str,
    ) -> List[tuple[str, str]]:
        """Build the selector model packet from marker-based extracts plus current user text."""
        segments: List[tuple[str, str]] = []
        seen: Set[str] = set()

        haystacks = [source_content, self._join_nonempty([source_content, context_content])]
        for idx, rule in enumerate(self.selector_context_rules):
            if not isinstance(rule, dict):
                continue
            name = str(rule.get("name") or f"selector_context_{idx}")
            if not self._rule_applies_to_request_type(rule, request_type, "selector_context", name):
                continue
            beginning = str(rule.get("beginning") or "")
            end = str(rule.get("end") or "")
            include_beginning_marker = bool(rule.get("include_beginning_marker", True))
            include_end_marker = bool(rule.get("include_end_marker", False))

            extracted = ""
            for haystack in haystacks:
                extracted = self._extract_selector_span(
                    haystack,
                    beginning=beginning,
                    end=end,
                    include_beginning_marker=include_beginning_marker,
                    include_end_marker=include_end_marker,
                )
                if extracted:
                    break
            extracted = self._strip_policy_regions_from_selector_context(extracted)
            extracted = self._clip_selector_context(extracted)
            if not extracted or extracted in seen:
                continue
            seen.add(extracted)
            segments.append((name, extracted))

        return segments

    def _compact_selector_title(self, title: str, limit: int = 96) -> str:
        """Keep selector title attributes short; summaries carry the real meaning."""
        cleaned = re.sub(r'\s+', ' ', str(title or '')).strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: max(0, limit - 3)].rstrip() + '...'

    def _section_selector_id(self, block_title: str, section: Any) -> str:
        """Stable ID used by the selector model to choose sections.

        When a prompt section matches the Static GM Index, expose that DB ID so
        selector logs and decisions are traceable back to source file/title/hash.
        """
        static_id = str(getattr(section, "static_index_id", "") or "").strip()
        if static_id:
            return static_id
        digest = hashlib.sha1(
            f"{block_title}\n{section.level}\n{section.title}\n{section.full_content}".encode("utf-8", errors="ignore")
        ).hexdigest()
        return f"sec_{digest[:12]}"

    def _extract_entities_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract all entities from text using the retriever's entity recognizer."""
        return self.retriever._extract_entities(text)

    def _iter_json_values(self, text: str):
        """Yield JSON values embedded inside arbitrary text."""
        decoder = json.JSONDecoder()
        i = 0
        while i < len(text):
            if text[i] not in '[{':
                i += 1
                continue
            try:
                value, end = decoder.raw_decode(text[i:])
                yield value
                i += max(end, 1)
            except json.JSONDecodeError:
                i += 1

    def _find_keyed_json_values(self, text: str, keys: Set[str]) -> List[Any]:
        """Find JSON values assigned to any of the given JSON-style keys."""
        results = []
        key_pattern = re.compile(r'"(' + '|'.join(re.escape(k) for k in keys) + r')"\s*:', re.IGNORECASE)
        decoder = json.JSONDecoder()

        for match in key_pattern.finditer(text):
            start = match.end()
            while start < len(text) and text[start].isspace():
                start += 1
            if start >= len(text) or text[start] not in '[{':
                continue
            try:
                value, _ = decoder.raw_decode(text[start:])
                results.append(value)
            except json.JSONDecodeError:
                continue
        return results

    def _extract_game_state(self, system_content: str) -> Optional[Dict[str, Any]]:
        """Extract game state from the system prompt if present."""
        game_state_keys = {'game_state', 'gamestate'}
        structural_keys = {'kingdoms', 'wars', 'alliances', 'active_wars', 'active_alliances', 'faction_list'}

        for value in self._find_keyed_json_values(system_content, game_state_keys):
            if isinstance(value, dict):
                return value

        for value in self._iter_json_values(system_content):
            if isinstance(value, dict):
                for key in game_state_keys:
                    nested = value.get(key)
                    if isinstance(nested, dict):
                        return nested
                if any(key in value for key in structural_keys):
                    return value
        return None

    def _filter_game_state(
        self,
        game_state: Dict[str, Any],
        entities: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Filter game state:
        - ALWAYS include: wars, alliances, kingdom list, dates
        - FILTER: army details, economic effects, resources to involved entities
        """
        filtered: Dict[str, Any] = {}

        always_include = [
            'wars', 'alliances', 'kingdoms', 'kingdom_list',
            'active_wars', 'active_alliances', 'faction_list',
            'current_date', 'game_date', 'year', 'season', 'day', 'turn',
        ]

        for key in always_include:
            if key in game_state:
                filtered[key] = game_state[key]

        entity_kingdoms = {str(k).lower() for k in entities.get('kingdoms', []) if k}
        entity_ids = {str(k).lower() for k in entities.get('string_ids', []) if k}
        relevant_entities = entity_kingdoms | entity_ids

        filter_keys = ['armies', 'economy', 'economic_effects', 'military_strength', 'resources']
        for key in filter_keys:
            if key not in game_state:
                continue
            value = game_state[key]
            if isinstance(value, dict):
                filtered_value = {
                    k: v for k, v in value.items()
                    if str(k).lower() in relevant_entities
                }
                if filtered_value:
                    filtered[key] = filtered_value
            elif isinstance(value, list):
                filtered_items = []
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    item_entities = {
                        str(item.get('kingdom', '')).lower(),
                        str(item.get('faction', '')).lower(),
                        str(item.get('faction_id', '')).lower(),
                        str(item.get('kingdom_id', '')).lower(),
                        str(item.get('owner', '')).lower(),
                    }
                    if item_entities & relevant_entities:
                        filtered_items.append(item)
                if filtered_items:
                    filtered[key] = filtered_items

        # Copy remaining small scalar fields as complete values, not character truncation.
        for key, value in game_state.items():
            if key in filtered or key in filter_keys:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                filtered[key] = value
            elif isinstance(value, dict) and len(value) <= 10:
                filtered[key] = value

        return filtered

    def _extract_event_history(self, system_content: str) -> Optional[List[Dict[str, Any]]]:
        """Extract event history from the system prompt if present."""
        keys = {'events', 'event_history', 'recent_events'}

        for value in self._find_keyed_json_values(system_content, keys):
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

        for value in self._iter_json_values(system_content):
            if isinstance(value, dict):
                for key in keys:
                    events = value.get(key)
                    if isinstance(events, list):
                        return [item for item in events if isinstance(item, dict)]
        return None

    def _extract_dialogue_history(self, system_content: str) -> Optional[List[Any]]:
        """Extract dialogue/conversation history from the system prompt if present."""
        keys = {'dialogue_history', 'conversation_history', 'recent_dialogue', 'chat_history'}

        for value in self._find_keyed_json_values(system_content, keys):
            if isinstance(value, list):
                return value

        for value in self._iter_json_values(system_content):
            if isinstance(value, dict):
                for key in keys:
                    dialogue = value.get(key)
                    if isinstance(dialogue, list):
                        return dialogue
        return None

    def _filter_event_history(
        self,
        events: List[Dict[str, Any]],
        entities: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Filter event history by recency and entity relevance.
        Keeps complete event objects.
        """
        if not events:
            return []

        entity_kingdoms = {str(k).lower() for k in entities.get('kingdoms', []) if k}
        entity_chars = {str(c).lower() for c in entities.get('characters', []) if c}

        scored_events = []
        total = len(events)
        for i, event in enumerate(events):
            # Newer events receive a higher baseline score.
            score = ((i + 1) / total) * 10
            event_str = json.dumps(event, ensure_ascii=False).lower()

            for kingdom in entity_kingdoms:
                if kingdom in event_str:
                    score += 5
            for char in entity_chars:
                if char in event_str:
                    score += 3

            scored_events.append((score, i, event))

        scored_events.sort(reverse=True, key=lambda x: x[0])
        selected = scored_events[:self.MAX_EVENT_HISTORY]
        selected.sort(key=lambda x: x[1])
        return [event for _, _, event in selected]


    def _is_policy_end_marker_line(self, line: str) -> bool:
        """True for standalone closing markers like [END GM] or === [END PINNED] ===."""
        return bool(re.match(
            r'^\s*(?:={3,}\s*)?\[END\s+(?:PINNED|PIN|GM|IGNORE)(?::[A-Z0-9_ -]+)*\](?:\s*={3,})?\s*\.?\s*$',
            str(line or "").strip(),
            re.IGNORECASE,
        ))

    def _normalize_section_title(self, title: str) -> str:
        """Normalize markdown-ish section titles for matching."""
        title = title.strip()
        title = re.sub(r'^(?:#{1,6}\s*|={2,}\s*)', '', title)
        title = re.sub(r'(?:\s*={2,}|#+)\s*$', '', title)
        title = re.sub(r'^\[(?:PINNED|PIN|GM|IGNORE)(?::[A-Z0-9_ -]+)*\]\s*', '', title.strip(), flags=re.IGNORECASE)
        title = title.replace('[', ' ').replace(']', ' ')
        title = re.sub(r'[^a-z0-9]+', ' ', title.lower()).strip()
        return title

    def _split_prompt_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Split AIInfluence's emitted prompt into complete markdown-ish sections.

        This is for hardcoded prompt text that we cannot change in the mod. It treats a section as
        a header plus all text until the next supported header. It does not cut a section by length.
        """
        header_re = re.compile(
            r'(?m)^(?P<header>'
            r'#{1,6}\s+.+?\s*$'
            r'|={3,}\s*.+?\s*={3,}\s*$'
            r'|##\s*={3,}\s*.+?\s*={3,}\s*$'
            r')'
        )
        matches = list(header_re.finditer(text))
        sections: List[Dict[str, str]] = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            header = match.group('header').strip()
            normalized = self._normalize_section_title(header)
            sections.append({
                'header': header,
                'title': normalized,
                'raw': text[start:end].strip(),
                'body': text[match.end():end].strip(),
            })
        return sections

    def _extract_aiinfluence_dynamic_sections(
        self,
        source_content: str,
        context_content: str,
        request_type: str,
    ) -> tuple[Dict[str, str], str]:
        """
        Extract/filter AIInfluence hardcoded dynamic data from the selected prompt-container.

        The mod is obfuscated, so we cannot make it emit ideal [PINNED]/[GM] markers. Instead,
        the proxy recognizes the stable section names that AIInfluence already emits and converts
        them into internal pseudo-sections. Filtering is conservative: exact/normalized matches
        first, fuzzy matches only for known IDs/names/items and only in lookup contexts.
        """
        sections = self._split_prompt_sections(source_content)
        by_title: Dict[str, List[str]] = {}
        for section in sections:
            by_title.setdefault(section['title'], []).append(section['raw'])

        filtered: Dict[str, str] = {}
        relevance_parts: List[str] = []
        focus_text = self._focus_text(context_content, context_content, source_content, request_type)

        # Dialogue prompt: current NPC/scene data. These are ground-truth facts, but some sublists
        # inside them can be large. Keep identity/personality/current situation; conditionally keep
        # money, item, nearby-person, settlement, and party detail.
        current_data = []

        for raw in by_title.get('global politics of the world', []):
            current_data.append(raw)
            relevance_parts.extend(self._extract_relevance_lines(raw))

        for raw in by_title.get('character briefing current data', []):
            slim = self._filter_character_briefing(raw, focus_text)
            if slim:
                current_data.append(slim)
                relevance_parts.extend(self._extract_relevance_lines(slim))

        for raw in by_title.get('immediate situation current data', []):
            current_data.append(raw)
            relevance_parts.extend(self._extract_relevance_lines(raw))

        for raw in by_title.get('the player current data', []):
            slim = self._filter_player_data(raw, focus_text)
            if slim:
                current_data.append(slim)
                relevance_parts.extend(self._extract_relevance_lines(slim))

        for raw in by_title.get('people physically present in this location right now', []):
            slim = self._filter_people_present(raw, focus_text)
            if slim:
                current_data.append(slim)
                relevance_parts.extend(self._extract_relevance_lines(slim))

        for raw in by_title.get('nearby settlements strategic context current data', []):
            slim = self._filter_nearby_list(raw, focus_text, kind='settlement')
            if slim:
                current_data.append(slim)
                relevance_parts.extend(self._extract_relevance_lines(slim))

        for raw in by_title.get('nearby parties npc vicinity current data', []):
            slim = self._filter_nearby_list(raw, focus_text, kind='party')
            if slim:
                current_data.append(slim)
                relevance_parts.extend(self._extract_relevance_lines(slim))

        for raw in by_title.get('mentioned settlements', []):
            slim = self._filter_nearby_list(raw, focus_text, kind='settlement')
            if slim:
                current_data.append(slim)
                relevance_parts.extend(self._extract_relevance_lines(slim))

        for raw in by_title.get('mentioned characters', []):
            slim = self._filter_mentioned_records(raw, focus_text, label='character')
            if slim:
                current_data.append(slim)
                relevance_parts.extend(self._extract_relevance_lines(slim))

        for raw in by_title.get('mentioned parties', []):
            slim = self._filter_nearby_list(raw, focus_text, kind='party')
            if slim:
                current_data.append(slim)
                relevance_parts.extend(self._extract_relevance_lines(slim))

        if current_data:
            filtered['ai_current_data'] = '\n\n'.join(current_data)

        # Dialogue prompt: conversation history. Keep last N whole speaker/metadata lines.
        conversation = []
        for raw in by_title.get('conversation history', []):
            slim = self._filter_plain_conversation_history(raw)
            if slim:
                conversation.append(slim)
                relevance_parts.extend(self._extract_relevance_lines(slim))
        if conversation:
            filtered['ai_conversation_history'] = '\n\n'.join(conversation)

        # Event/diplomacy prompt: current world state is ground truth and should stay complete.
        world_state = []
        for title in ('current world data ground truth', 'current world state'):
            for raw in by_title.get(title, []):
                world_state.append(raw)
                relevance_parts.extend(self._extract_relevance_lines(raw))
        if world_state:
            filtered['ai_current_world_state'] = '\n\n'.join(world_state)

        # Event/diplomacy prompt: historical events are for avoiding duplicates. Keep complete events,
        # capped by max_event_history.
        existing_events = []
        for title in ('existing events historical narrative', 'existing dynamic events do not duplicate'):
            for raw in by_title.get(title, []):
                slim = self._filter_bulleted_entries(raw, max_entries=self.MAX_EVENT_HISTORY)
                if slim:
                    existing_events.append(slim)
                    relevance_parts.extend(self._extract_relevance_lines(slim))
        if existing_events:
            filtered['ai_existing_events'] = '\n\n'.join(existing_events)

        econ = []
        for raw in by_title.get('active economic effects', []):
            econ.append(raw)
            relevance_parts.extend(self._extract_relevance_lines(raw))
        if econ:
            filtered['ai_active_economic_effects'] = '\n\n'.join(econ)

        diplomacy = []
        for raw in by_title.get('recent diplomatic statements last 15 statements from last 50 days', []):
            slim = self._filter_recent_diplomatic_statements(raw)
            if slim:
                diplomacy.append(slim)
                relevance_parts.extend(self._extract_relevance_lines(slim))
        if diplomacy:
            filtered['ai_recent_diplomatic_statements'] = '\n\n'.join(diplomacy)

        new_dialogues = []
        for raw in by_title.get('new npc dialogues since last analysis', []):
            slim = self._filter_new_npc_dialogues(raw, context_content=focus_text)
            if slim:
                new_dialogues.append(slim)
                relevance_parts.extend(self._extract_relevance_lines(slim))
        if new_dialogues:
            filtered['ai_new_npc_dialogues'] = '\n\n'.join(new_dialogues)

        return filtered, self._join_nonempty(relevance_parts)

    def _focus_text(self, context_content: str, dialogue_query: str, source_content: str, request_type: str) -> str:
        """Text used only for relevance decisions; never sent by itself."""
        if request_type == "dialogue" and dialogue_query.strip():
            return dialogue_query.strip()
        return self._extract_request_focus(source_content, request_type)

    def _norm_lookup(self, text: str) -> str:
        text = text.lower().replace('_', ' ')
        return re.sub(r'[^a-z0-9]+', ' ', text).strip()

    def _lookup_tokens(self, text: str) -> List[str]:
        stop = {
            'the','and','or','to','for','from','with','without','about','into','onto','near','here','there',
            'this','that','these','those','please','would','could','should','will','can','you','your','have',
            'has','him','her','his','she','he','they','them','what','where','when','why','how','want','need',
            'give','take','tell','ask','say','said','pay','paid','gold','denars','denar','money','service','services'
        }
        toks = [t for t in self._norm_lookup(text).split() if len(t) >= 3 and t not in stop]
        return toks

    def _has_any_token(self, text: str, tokens: List[str]) -> bool:
        hay = set(self._norm_lookup(text).split())
        return any(tok in hay for tok in tokens)

    def _contains_context(self, query: str, keywords: Iterable[str]) -> bool:
        q = self._norm_lookup(query)
        return any(re.search(r'\b' + re.escape(self._norm_lookup(k)) + r'\b', q) for k in keywords)

    def _extract_line_candidates(self, line: str) -> List[str]:
        """Known lookup candidates from a data line: IDs, display names before (id:...), and item names."""
        candidates: List[str] = []
        for m in re.finditer(r'\bid:([A-Za-z0-9_\-]+)', line):
            candidates.append(m.group(1))
        # "Name (id:foo)" pattern.
        for m in re.finditer(r'([A-Z][A-Za-z\'’\-]+(?:\s+[A-Z][A-Za-z\'’\-]+){0,3})\s*\(id:', line):
            candidates.append(m.group(1))
        # Item pattern: "Grain (id:grain): 459" inside comma-separated inventory lines.
        for m in re.finditer(r'([A-Za-z][A-Za-z\'’\- ]{1,40})\s*\(id:[A-Za-z0-9_\-]+\)\s*:', line):
            candidates.append(m.group(1).strip())
        # Settlement line: "- Pennytree (id:village...)".
        first = re.match(r'\s*-\s*([^,:(]{3,80})\s*\(id:', line)
        if first:
            candidates.append(first.group(1).strip())
        # Dedup normalized.
        out, seen = [], set()
        for c in candidates:
            n = self._norm_lookup(c)
            if len(n) >= 3 and n not in seen:
                seen.add(n)
                out.append(c)
        return out

    def _fuzzy_candidate_in_query(self, candidate: str, query: str, *, allow_single_token: bool = False) -> bool:
        """
        Conservative typo tolerance. It never rewrites words; it only decides whether a known record
        should be included. Single-token fuzzy matches are intentionally strict to avoid power→powder.
        """
        cand = self._norm_lookup(candidate)
        query_norm = self._norm_lookup(query)
        if not cand or not query_norm:
            return False
        if re.search(r'\b' + re.escape(cand) + r'\b', query_norm):
            return True

        cand_tokens = cand.split()
        q_tokens = query_norm.split()
        if not cand_tokens or not q_tokens:
            return False

        # Multi-word names tolerate transposed/missing letters: "jhon snow" ~ "jon snow".
        if len(cand_tokens) >= 2:
            n = len(cand_tokens)
            for i in range(0, max(len(q_tokens) - n + 1, 0)):
                phrase = ' '.join(q_tokens[i:i+n])
                if SequenceMatcher(None, cand, phrase).ratio() >= max(0.86, self.fuzzy_match_threshold):
                    return True
            return False

        # Single-token fuzzy is for known IDs/items only and requires explicit lookup context elsewhere.
        if not allow_single_token or len(cand) < 5:
            return False
        for token in q_tokens:
            if len(token) < 5:
                continue
            # Same first two characters + very high ratio prevents normal words like power matching powder.
            if cand[:2] == token[:2] and SequenceMatcher(None, cand, token).ratio() >= 0.94:
                return True
        return False

    def _line_is_relevant(self, line: str, query: str, *, fuzzy_context: bool = False, allow_single_token: bool = False) -> bool:
        tokens = self._lookup_tokens(query)
        if self._has_any_token(line, tokens):
            return True
        if not fuzzy_context:
            return False
        for cand in self._extract_line_candidates(line):
            if self._fuzzy_candidate_in_query(cand, query, allow_single_token=allow_single_token):
                return True
        return False

    def _filter_player_data(self, raw: str, query: str) -> str:
        """Filter player data: keep identity/relationship facts, hide equipment/item IDs unless relevant."""
        if not self.dynamic_filter_enabled:
            return raw.strip()
        lines = raw.splitlines()
        if not lines:
            return ''
        header = lines[0]
        out: List[str] = [header]

        appearance_ctx = self._contains_context(query, [
            'appearance','look','looks','wear','wearing','armor','armour','helmet','boots','weapon','weapons',
            'sword','shield','pitchfork','equipment','gear','clothes','clothing','disguise','recognize'
        ])
        money_ctx = self._contains_context(query, ['denar','denars','gold','money','pay','payment','reward','ransom','hire','bribe','price','cost','buy','sell','trade','barter'])
        item_ctx = appearance_ctx or self._contains_context(query, [
            'grain','food','beer','meat','cheese','butter','horse','horses','mount','livestock','tools','linen',
            'jewelry','silver','ore','silk','flax','felt','wool','hog','sheep','item','items','inventory','trade',
            'barter','buy','sell','give','take','spare'
        ])

        for line in lines[1:]:
            stripped = line.strip()
            if stripped.startswith('- **Their Appearance'):
                if appearance_ctx or item_ctx:
                    out.append(line)  # keep full IDs because relevant gear/equipment may be actionable
                else:
                    out.append('- **Their Appearance:** Equipment details hidden by proxy until appearance/equipment/items are relevant.')
                continue
            if stripped.startswith('- **Their Inventory') or stripped.startswith('- **Player') and 'Inventory' in stripped:
                if item_ctx:
                    out.append(line)
                else:
                    out.append('- **Player Inventory:** Hidden by proxy until item/trade/barter is relevant.')
                continue
            if stripped.startswith('- **Their Wealth') or stripped.startswith('- **Player Wealth'):
                if money_ctx:
                    out.append(line)
                else:
                    out.append('- **Player Wealth:** Hidden by proxy until money/payment/trade is relevant.')
                continue
            out.append(line)
        return '\n'.join(out).strip()

    def _briefing_line_context(self, line: str) -> str:
        s = line.strip().lower()
        if s.startswith('- **relatives'):
            return 'family'
        if s.startswith('- **friends') or s.startswith('- **friends & enemies'):
            return 'relations'
        if s.startswith('- **your forces') or s.startswith('- **forces'):
            return 'forces'
        if s.startswith('- **your captives') or s.startswith('- **captives'):
            return 'captives'
        if s.startswith('- **your workshops') or s.startswith('- **workshops'):
            return 'workshops'
        return ''

    def _filter_character_briefing(self, raw: str, query: str) -> str:
        """Keep character essentials; conditionally keep wealth, inventory, and large clan lists."""
        if not self.dynamic_filter_enabled:
            return raw.strip()
        lines = raw.splitlines()
        if not lines:
            return ''
        header = lines[0]
        out: List[str] = [header]

        money_ctx = self._contains_context(query, ['denar','denars','gold','money','pay','payment','reward','ransom','hire','bribe','price','cost','buy','sell','trade','barter','services'])
        item_ctx = self._contains_context(query, ['grain','food','beer','meat','cheese','butter','horse','horses','mount','livestock','tools','linen','jewelry','silver','ore','silk','flax','felt','wool','hog','sheep','item','items','inventory','trade','barter','buy','sell','give me','spare'])
        clan_ctx = self._contains_context(query, ['family','father','mother','wife','husband','spouse','son','daughter','brother','sister','clan','relative','kin','member','heir','leader'])
        relation_ctx = self._contains_context(query, ['friend','friends','enemy','enemies','rival','ally','allies','relation','relations','trust','feud'])
        forces_ctx = self._contains_context(query, ['army','forces','soldiers','troops','party','strength','battle','war','siege','attack','defend','patrol','march','reinforce'])
        captive_ctx = self._contains_context(query, ['captive','captives','prisoner','prisoners','ransom','release','execute','captured'])
        appearance_ctx = self._contains_context(query, ['appearance','look','looks','wear','wearing','armor','armour','helmet','boots','weapon','weapons','equipment','gear','clothes','clothing','recognize'])
        workshop_ctx = self._contains_context(query, ['workshop','shop','business','sell workshop','buy workshop','income'])

        in_inventory = False
        inventory_lines: List[str] = []
        for line in lines[1:]:
            stripped = line.strip()
            if stripped.startswith('- **Your Inventory'):
                in_inventory = True
                inventory_lines.append(line)
                continue
            if in_inventory:
                # Inventory block ends at next top-level briefing bullet.
                if stripped.startswith('- **') and not stripped.startswith('- **Your Inventory'):
                    slim_inv = self._filter_inventory_block(inventory_lines, query, include_any=item_ctx)
                    if slim_inv:
                        out.extend(slim_inv)
                    inventory_lines = []
                    in_inventory = False
                    # fall through and process this new non-inventory line.
                else:
                    inventory_lines.append(line)
                    continue

            if stripped.startswith('- **Your Wealth'):
                if money_ctx:
                    out.append(line)
                else:
                    out.append('- **Your Wealth:** Available if money/payment/trade becomes relevant.')
                continue

            if stripped.startswith('- **Appearance'):
                if appearance_ctx or item_ctx:
                    out.append(line)  # keep IDs when appearance/equipment is relevant
                else:
                    out.append('- **Appearance:** Gear/equipment details hidden by proxy until appearance/equipment/items are relevant.')
                continue

            if stripped.startswith('- **Clan:**'):
                if clan_ctx or self._line_is_relevant(line, query, fuzzy_context=True):
                    out.append(line)
                else:
                    # Keep current clan identity/holdings/wars but remove the long all-members list.
                    clan_id = re.search(r'You are .*?\.\s*', line)
                    holdings = re.search(r'Clan holdings:.*?(?:\.\s*|$)', line)
                    wars = re.search(r'Clan wars:.*?(?:\.\s*|$)', line)
                    parts = [p.group(0).strip() for p in (clan_id, holdings, wars) if p]
                    out.append('- **Clan:** ' + (' '.join(parts) if parts else 'Clan details available if clan/family becomes relevant.'))
                continue

            line_context = self._briefing_line_context(line)
            if line_context == 'family':
                if clan_ctx or self._line_is_relevant(line, query, fuzzy_context=True):
                    out.append(line)
                else:
                    out.append('- **Relatives:** Hidden by proxy until family/relative/person is relevant.')
                continue
            if line_context == 'relations':
                if relation_ctx or self._line_is_relevant(line, query, fuzzy_context=True):
                    out.append(line)
                else:
                    out.append('- **Friends & Enemies:** Hidden by proxy until relations/specific people are relevant.')
                continue
            if line_context == 'forces':
                if forces_ctx or self._line_is_relevant(line, query, fuzzy_context=True):
                    out.append(line)
                else:
                    out.append('- **Your Forces:** Detailed forces hidden by proxy until military/troop/action context is relevant.')
                continue
            if line_context == 'captives':
                if captive_ctx or self._line_is_relevant(line, query, fuzzy_context=True):
                    out.append(line)
                else:
                    out.append('- **Your Captives:** Hidden by proxy until prisoners/ransom/release are relevant.')
                continue
            if line_context == 'workshops':
                if workshop_ctx or money_ctx or self._line_is_relevant(line, query, fuzzy_context=True):
                    out.append(line)
                else:
                    out.append('- **Workshops:** Hidden by proxy until workshop/business/trade is relevant.')
                continue

            # Keep ordinary identity/personality/current-status fields.
            out.append(line)

        if in_inventory:
            slim_inv = self._filter_inventory_block(inventory_lines, query, include_any=item_ctx)
            if slim_inv:
                out.extend(slim_inv)

        return '\n'.join(out).strip()

    def _filter_inventory_block(self, lines: List[str], query: str, *, include_any: bool) -> List[str]:
        if not lines:
            return []
        header = lines[0]
        note_lines = [l for l in lines[1:] if l.strip().startswith('**NOTE') or l.strip().startswith('(Use exact')]
        item_lines = [l for l in lines[1:] if re.match(r'\s*-\s+[^:]+:', l)]
        if not include_any:
            return [header, '(Inventory hidden by proxy until item/trade/barter is relevant.)']

        selected: List[str] = []
        # Exact category/item token matches first; fuzzy only in item/trade context and only against known item names/IDs.
        for line in item_lines:
            if self._line_is_relevant(line, query, fuzzy_context=True, allow_single_token=True):
                selected.append(self._filter_inventory_category_line(line, query))
        if not selected and self._contains_context(query, ['food','supplies','provisions']):
            selected = [l for l in item_lines if re.search(r'\bFood\s*:', l, re.IGNORECASE)]
        if not selected and self._contains_context(query, ['horse','horses','mount','mounts','livestock']):
            selected = [l for l in item_lines if re.search(r'\bMounts/Livestock\s*:', l, re.IGNORECASE)]
        if not selected and self._contains_context(query, ['trade','barter','inventory','items','goods']):
            selected = item_lines[:self.max_inventory_lines]

        return [header] + note_lines[:1] + selected[:self.max_inventory_lines] + ['(Inventory filtered by proxy to relevant categories/items.)']

    def _filter_inventory_category_line(self, line: str, query: str) -> str:
        # Keep only matching comma-separated items where possible, preserving category label.
        m = re.match(r'(\s*-\s*[^:]+:\s*)(.*)$', line)
        if not m:
            return line
        prefix, rest = m.group(1), m.group(2)
        items = [x.strip() for x in rest.split(',') if x.strip()]
        selected = [item for item in items if self._line_is_relevant(item, query, fuzzy_context=True, allow_single_token=True)]
        if not selected:
            return line
        return prefix + ', '.join(selected)

    def _filter_people_present(self, raw: str, query: str) -> str:
        if not self.dynamic_filter_enabled:
            return raw.strip()
        people_ctx = self._contains_context(query, ['who is here','who\'s here','present','nearby','escort','walk over','talk to','speak to','follow','bring','find','lord','lady','person','people','someone'])
        lines = raw.splitlines()
        if not lines:
            return ''
        header = lines[0]
        intro = [l for l in lines[1:] if l.strip().startswith('(')]
        entries = [l for l in lines[1:] if re.match(r'\s*-\s+', l)]
        selected = [l for l in entries if self._line_is_relevant(l, query, fuzzy_context=True)]
        if not selected and people_ctx:
            selected = entries[:self.max_people_present]
        if not selected:
            return '\n'.join([header] + intro + ['(People-present list hidden by proxy until a nearby-person/action/name is relevant.)']).strip()
        more = len(entries) - len(selected)
        suffix = [f'({more} other people present hidden by proxy.)'] if more > 0 else []
        return '\n'.join([header] + intro + selected[:self.max_people_present] + suffix).strip()

    def _filter_mentioned_records(self, raw: str, query: str, *, label: str) -> str:
        if not self.dynamic_filter_enabled:
            return raw.strip()
        lines = raw.splitlines()
        if not lines:
            return ''
        header = lines[0]
        intro = [l for l in lines[1:] if l.strip().startswith('(')]
        entries = [l for l in lines[1:] if re.match(r'\s*-\s+', l)]
        ctx = self._contains_context(query, [label, label+'s', 'who', 'name', 'id', 'talk', 'speak', 'find', 'relation', 'family', 'friend', 'enemy'])
        selected = [l for l in entries if self._line_is_relevant(l, query, fuzzy_context=True)]
        if not selected and ctx:
            selected = entries[:self.max_people_present]
        if not selected:
            return '\n'.join([header] + intro + [f'({label.title()} records hidden by proxy until relevant.)']).strip()
        more = len(entries) - len(selected)
        suffix = [f'({more} other {label} records hidden by proxy.)'] if more > 0 else []
        return '\n'.join([header] + intro + selected[:self.max_people_present] + suffix).strip()

    def _filter_nearby_list(self, raw: str, query: str, *, kind: str) -> str:
        if not self.dynamic_filter_enabled:
            return raw.strip()
        if kind == 'settlement':
            ctx_words = ['where','location','settlement','town','castle','village','travel','go to','patrol','raid','siege','nearby','distance','property','attack','defend']
            max_entries = self.max_nearby_settlements
        else:
            ctx_words = ['party','parties','army','nearby','villagers','caravan','bandit','enemy','ally','attack','escort','follow','distance','target','war']
            max_entries = self.max_nearby_parties
        list_ctx = self._contains_context(query, ctx_words)
        lines = raw.splitlines()
        if not lines:
            return ''
        header = lines[0]
        intro = [l for l in lines[1:] if l.strip().startswith('(')]
        entries = [l for l in lines[1:] if re.match(r'\s*-\s+', l)]
        selected = [l for l in entries if self._line_is_relevant(l, query, fuzzy_context=True)]
        if not selected and list_ctx:
            selected = entries[:max_entries]
        if not selected:
            label = 'settlement/location' if kind == 'settlement' else 'nearby-party'
            return '\n'.join([header] + intro + [f'({label} details hidden by proxy until relevant.)']).strip()
        more = len(entries) - len(selected)
        suffix = [f'({more} other {kind} entries hidden by proxy.)'] if more > 0 else []
        return '\n'.join([header] + intro + selected[:max_entries] + suffix).strip()

    def _strip_incoming_world_blocks_from_text(self, text: str) -> str:
        """Remove full ### The World ### blocks from a prompt while preserving following sections."""
        lines = text.splitlines()
        out: List[str] = []
        i = 0
        while i < len(lines):
            if self._normalize_section_title(lines[i]) == 'the world':
                i += 1
                while i < len(lines):
                    title = self._normalize_section_title(lines[i])
                    if title in {'global politics of the world', 'character briefing current data'}:
                        break
                    i += 1
                continue
            out.append(lines[i])
            i += 1
        return '\n'.join(out)

    def _extract_incoming_world_blocks(self, text: str) -> List[str]:
        """Extract full ### The World ### blocks, including internal world headers, from the incoming prompt."""
        lines = text.splitlines()
        blocks: List[str] = []
        i = 0
        while i < len(lines):
            if self._normalize_section_title(lines[i]) == 'the world':
                start = i
                i += 1
                while i < len(lines):
                    title = self._normalize_section_title(lines[i])
                    # In dialogue prompts this is the stable section after The World. Do not stop on
                    # internal world.txt headers like === [GM] Cultures === or nested ## entries.
                    if title in {'global politics of the world', 'character briefing current data'}:
                        break
                    i += 1
                blocks.append('\n'.join(lines[start:i]).strip())
                continue
            i += 1
        return blocks

    def _filter_incoming_world_blocks(self, blocks: List[str], selection_query: str, entities: Dict[str, List[str]], request_type: str = "dialogue") -> tuple[str, int, int]:
        """Apply [PINNED]/[GM]/[IGNORE] to the world.txt content already injected by AIInfluence."""
        rendered_blocks: List[str] = []
        total_selected = 0
        total_pinned = 0
        for block in blocks:
            lines = block.splitlines()
            header = lines[0] if lines else '### The World ###'
            body = '\n'.join(lines[1:]).strip()
            if not body:
                continue
            sections = self.retriever._parse_sections(body, 'incoming_world', request_type=request_type)
            self._promote_inferred_structured_top_level_world_sections(sections)
            pinned = []
            gm_candidates = []
            for sec in sections:
                if sec.policy == self.retriever.POLICY_IGNORE:
                    continue
                if self.retriever._is_empty_container_section(sec):
                    continue
                sec.entities = self.retriever._extract_entities(sec.full_content)
                sec.summary = self.retriever._generate_summary(sec)
                if sec.policy == self.retriever.POLICY_PINNED:
                    pinned.append(sec)
                else:
                    gm_candidates.append(sec)
            selected = self._select_incoming_world_gm_sections(gm_candidates, selection_query, entities)
            total_selected += len(selected)
            total_pinned += len(pinned)
            rendered = [header]
            selected_ids = {id(sec) for sec in selected}
            body_rendered, _ = self._render_selected_world_sections(sections, selected_ids)
            rendered.extend(body_rendered)
            rendered_blocks.append('\n\n'.join(rendered).strip())
        return self._normalize_render_spacing('\n\n'.join(rendered_blocks)), total_selected, total_pinned

    def _render_selected_world_sections(self, sections: List[Any], selected_ids: Set[int]) -> tuple[List[str], int]:
        """Render included sections in order, always preserving parent headers.

        The important invariant is: a selected ``##`` child must never be dumped into the
        final prompt as a bare child list.  Compact policy headers made the old logic too
        dependent on whether the empty parent container survived a particular parse path.
        This renderer therefore emits a parent header before the first included child of
        each parent, using the authored parent line when present and the static-index parent
        title as fallback.
        """
        parent_with_selected_children: Set[int] = set()
        current_parent = None
        for sec in sections:
            if sec.policy == self.retriever.POLICY_IGNORE:
                continue
            if sec.level == 1:
                current_parent = sec
                continue
            if id(sec) in selected_ids and current_parent is not None:
                parent_with_selected_children.add(id(current_parent))

        rendered: List[str] = []
        selected_count = 0
        current_parent = None
        rendered_parent_ids: Set[int] = set()
        rendered_parent_keys: Set[str] = set()

        def mark_parent(parent: Any, header: str) -> None:
            if parent is not None:
                rendered_parent_ids.add(id(parent))
            key = self._parent_render_key(header)
            if key:
                rendered_parent_keys.add(key)

        def ensure_parent_for_child(child: Any, parent: Any = None) -> None:
            header = self._parent_header_for_child(child, parent)
            key = self._parent_render_key(header)
            if header and key not in rendered_parent_keys:
                rendered.append(header)
                mark_parent(parent, header)

        for sec in sections:
            if sec.policy == self.retriever.POLICY_IGNORE:
                continue

            if sec.level == 1:
                current_parent = sec
                include_parent = (
                    sec.policy == self.retriever.POLICY_PINNED
                    or id(sec) in selected_ids
                    or id(sec) in parent_with_selected_children
                )
                if include_parent:
                    if id(sec) in parent_with_selected_children and sec.policy != self.retriever.POLICY_PINNED and id(sec) not in selected_ids:
                        block = self._render_header_only_for_section(sec)
                        # Do not render separator junk as a parent title.  The selected child
                        # will still force a DB-authored parent header below.
                        if not self._meaningful_parent_header(block):
                            block = ""
                    else:
                        block = self._format_world_section(sec.title, sec.full_content)
                    key = self._parent_render_key(block)
                    if block and key not in rendered_parent_keys:
                        rendered.append(block)
                        mark_parent(sec, block)
                    if id(sec) in selected_ids:
                        selected_count += 1
                        state = self._current_static_selection_state()
                        if state is not None:
                            sid = str(getattr(sec, "static_index_id", "") or "").strip()
                            if sid:
                                state.rendered_ids.add(sid)
                continue

            if self.retriever._is_empty_container_section(sec):
                continue
            include = sec.policy == self.retriever.POLICY_PINNED or id(sec) in selected_ids
            if not include:
                continue

            # Always ensure a parent header exists before a selected child.  This uses the
            # live parsed parent when available; otherwise it falls back to the DB parent
            # stored on the child.  This is what fixes compact headers such as
            # [GM:DIALOGUE] ### Cultures ###.
            ensure_parent_for_child(sec, current_parent)

            if id(sec) in selected_ids:
                selected_count += 1
                state = self._current_static_selection_state()
                if state is not None:
                    sid = str(getattr(sec, "static_index_id", "") or "").strip()
                    if sid:
                        state.rendered_ids.add(sid)
            rendered.append(self._format_world_section(sec.title, sec.full_content))

        return rendered, selected_count

    async def _select_via_selector_model(
        self,
        sections: List[Any],
        request_type: str,
        dialogue_query: str,
        context_segments: List[tuple[str, str]],
        block_title: str,
        fallback_query: str,
        fallback_entities: Dict[str, List[str]],
    ) -> List[Any]:
        """Use the configured selector model for [GM] decisions."""
        if not sections or not self.selector_client or not self.selector_client.enabled:
            return []

        grouped_sections = self._group_selector_candidate_sections(sections, block_title)
        block_payloads: List[Dict[str, Any]] = []
        alias_counter = 1
        for index, (visible_block_title, group_sections) in enumerate(grouped_sections, start=1):
            candidates: List[Dict[str, str]] = []
            for section in group_sections:
                # The selector only needs a request-local handle it can return.  The
                # long static DB id is kept internally for traceability/injection, but
                # it is not sent to the middleman because it can contain a huge slug.
                real_section_id = self._section_selector_id(visible_block_title, section)
                selector_alias = f"s{alias_counter:03d}"
                alias_counter += 1
                setattr(section, "selector_alias_id", selector_alias)
                setattr(section, "selector_real_id", real_section_id)

                rendered = self._format_world_section(section.title, section.full_content)
                summary = str(getattr(section, "selector_summary", "") or getattr(section, "summary", "") or "").strip()
                candidate: Dict[str, str] = {
                    "id": selector_alias,
                    "title": self._compact_selector_title(str(section.title)),
                }
                if self._static_gm_index_available() and self.static_gm_index_selector_payload == "summary":
                    # Static-index selector mode must never fall back to full section bodies.
                    # If the DB/LLM summary is missing, expose a clipped deterministic
                    # selector summary instead of leaking the whole [GM] child to the middleman.
                    if not summary:
                        summary = self.retriever._generate_summary(section)
                    candidate.update({
                        "summary": self._clip_selector_summary(summary),
                    })
                else:
                    candidate["content"] = rendered
                candidates.append(candidate)
            block_payloads.append({
                "id": f"block_{index}",
                "title": visible_block_title,
                "sections": candidates,
                "source_sections": group_sections,
            })

        try:
            keep_ids_by_block = await self.selector_client.select_relevant_blocks(
                request_type=request_type,
                dialogue_query=dialogue_query,
                context_segments=context_segments,
                blocks=[
                    {
                        "id": block["id"],
                        "title": block["title"],
                        "sections": block["sections"],
                    }
                    for block in block_payloads
                ],
            )
        except Exception as exc:
            # Fail closed. Keeping all GM candidates on selector/API/JSON failure can explode
            # the main-model prompt into tens or hundreds of thousands of tokens. Pinned
            # sections are still rendered by the caller; GM sections are omitted unless the
            # selector explicitly returns their request-local aliases.
            logger.error("Selector model failed; fail-closed with zero GM selections: %s", exc)
            if self.selector_client:
                try:
                    await self.selector_client.log_diagnostic(
                        "SELECTOR FAILURE FAIL-CLOSED",
                        {
                            "request_type": request_type,
                            "error": str(exc),
                            "candidate_blocks": [
                                {
                                    "id": str(block.get("id", "") or ""),
                                    "title": str(block.get("title", "") or ""),
                                    "candidate_count": len(block.get("sections") or []),
                                }
                                for block in block_payloads
                            ],
                        },
                    )
                except Exception:
                    pass
            return []

        selected_seen: Set[int] = set()
        for block in block_payloads:
            visible_block_title = str(block["title"])
            group_sections = list(block["source_sections"])
            keep_set = set(keep_ids_by_block.get(str(block["id"]), []))
            for section in group_sections:
                section_id = str(getattr(section, "selector_alias_id", "") or self._section_selector_id(visible_block_title, section))
                if section_id not in keep_set:
                    continue
                selected_seen.add(id(section))

        return [section for section in sections if id(section) in selected_seen]

    def _normalize_selector_block_title(self, block_title: str, sections: List[Any]) -> str:
        """Choose a human-readable block title for selector prompts."""
        cleaned = str(block_title or "").strip()
        if cleaned:
            return cleaned
        for section in sections:
            if getattr(section, "level", 0) == 1 and getattr(section, "title", ""):
                return str(section.title).strip()
        return "GM Block"

    def _group_selector_candidate_sections(self, sections: List[Any], fallback_block_title: str) -> List[tuple[str, List[Any]]]:
        """Split selector candidates into one group per top-level [GM] parent.

        Static-index preparation can omit the empty parent container from the candidate
        list, so each child carries selector_parent_title metadata.  Use that metadata to
        keep children grouped under their original parent instead of flattening them into
        one block per child.
        """
        if not sections:
            return []

        # Static-index path: children already know their original parent title.
        if any(str(getattr(section, "selector_parent_title", "") or "").strip() for section in sections):
            groups_by_title: Dict[str, List[Any]] = {}
            order: List[str] = []
            for section in sections:
                if self.retriever._is_empty_container_section(section):
                    continue
                parent_title = str(getattr(section, "selector_parent_title", "") or "").strip()
                title = str(getattr(section, "title", "") or "").strip()
                group_title = parent_title or self._normalize_selector_block_title(fallback_block_title, [section]) or title or "GM Block"
                if group_title not in groups_by_title:
                    groups_by_title[group_title] = []
                    order.append(group_title)
                groups_by_title[group_title].append(section)
            return [(title, groups_by_title[title]) for title in order if groups_by_title[title]]

        groups: List[tuple[str, List[Any]]] = []
        current_group_title = ""
        current_group_sections: List[Any] = []

        def flush_group() -> None:
            nonlocal current_group_title, current_group_sections
            if not current_group_sections:
                return
            groups.append((current_group_title or self._normalize_selector_block_title(fallback_block_title, current_group_sections), list(current_group_sections)))
            current_group_title = ""
            current_group_sections = []

        for section in sections:
            level = int(getattr(section, "level", 0) or 0)
            title = str(getattr(section, "title", "") or "").strip()
            is_empty = self.retriever._is_empty_container_section(section)

            if level == 1:
                flush_group()
                current_group_title = title or self._normalize_selector_block_title(fallback_block_title, [section])
                current_group_sections = [] if is_empty else [section]
                continue

            if current_group_title:
                current_group_sections.append(section)
                continue

            groups.append((title or self._normalize_selector_block_title(fallback_block_title, [section]), [section]))

        flush_group()
        return groups

    def _is_low_information_query(self, query: str) -> bool:
        tokens = self._lookup_tokens(query)
        return len(tokens) < 2

    def _select_incoming_world_gm_sections(self, sections: List[Any], query: str, entities: Dict[str, List[str]]) -> List[Any]:
        if not sections:
            return []
        logger.error("Legacy direct-selection path called; fail-closed with zero GM selections")
        return []

    def _extract_relevance_lines(self, text: str) -> List[str]:
        """
        Extract complete short-ish lines that are good selector signals.
        This affects only selector-side matching, not what gets sent to the backend.
        """
        signal_re = re.compile(
            r'\b(id:|string_id|culture|kingdom|faction|clan|settlement|location|war|alliance|peace|'
            r'siege|captur|army|leader|ruler|spouse|lord|lady|current task|recent events|diplomatic)\b',
            re.IGNORECASE,
        )
        lines: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and signal_re.search(stripped):
                lines.append(stripped)
        # Keep complete lines, but cap the number of relevance lines so selector matching is not polluted.
        return lines[:80]

    def _filter_plain_conversation_history(self, raw: str) -> str:
        """Keep the last N whole dialogue-history lines from an AIInfluence conversation section."""
        lines = [line for line in raw.splitlines() if line.strip()]
        if not lines:
            return ''
        header = lines[0]
        body = lines[1:]
        if self.DIALOGUE_HISTORY_SIZE <= 0:
            return header
        speaker_re = re.compile(r'^\s*(Player|[^:]{1,80}):')
        meta_re = re.compile(r'^\s*(Last Interaction|Previous Response|NPC|Character|System):', re.IGNORECASE)
        entries = [line for line in body if speaker_re.match(line) or meta_re.match(line)]
        selected = entries[-self.DIALOGUE_HISTORY_SIZE:] if entries else body[-self.DIALOGUE_HISTORY_SIZE:]
        return '\n'.join([header] + selected)

    def _filter_bulleted_entries(self, raw: str, max_entries: int) -> str:
        """Keep the last N complete bullet/list entries from a section."""
        lines = raw.splitlines()
        if not lines:
            return ''
        header = lines[0]
        entries: List[List[str]] = []
        current: List[str] = []
        for line in lines[1:]:
            if re.match(r'^\s*-\s+', line):
                if current:
                    entries.append(current)
                current = [line]
            elif current:
                current.append(line)
            elif line.strip():
                # Non-bulleted intro text.
                current = [line]
        if current:
            entries.append(current)
        if not entries:
            return raw
        selected = entries[-max_entries:] if max_entries > 0 else []
        return '\n'.join([header] + ['\n'.join(entry) for entry in selected]).strip()

    def _filter_recent_diplomatic_statements(self, raw: str) -> str:
        """Keep complete recent diplomatic statement blocks. This section is usually small."""
        # It is already capped by the mod ("Last 15 statements"), so preserving it is safe.
        return raw.strip()


    def _strip_dialogue_technical_appendices(self, convo: str) -> str:
        """Return only the spoken dialogue portion of a New dialogue block for scoring."""
        cut_markers = [
            '\n  **Settlements mentioned',
            '\n  **Characters mentioned',
            '\n  **Parties mentioned',
            '\n  **Relevant Current Data',
            '\n  **Location context',
            '\n  **Current context',
        ]
        result = convo
        for marker in cut_markers:
            pos = result.find(marker)
            if pos != -1:
                result = result[:pos]
        return result

    def _dialogue_name_tokens(self, text: str) -> Set[str]:
        """Extract lowercase name-like tokens from dialogue text for appendix filtering."""
        tokens: Set[str] = set()
        for match in re.finditer(r"\b[A-Z][A-Za-z'’.-]{2,}(?:\s+[A-Z][A-Za-z'’.-]{2,}){0,3}\b", text):
            phrase = re.sub(r'\s+', ' ', match.group(0)).strip().lower()
            if phrase:
                tokens.add(phrase)
                for part in phrase.split():
                    if len(part) >= 4:
                        tokens.add(part)
        return tokens

    def _settlement_entry_name(self, entry: str) -> str:
        m = re.match(r'^\s*-\s+(.+?)\s+\(id:', entry)
        return (m.group(1).strip() if m else entry.strip()).lower()

    def _entry_matches_dialogue_text(self, entry: str, dialogue_text: str, tokens: Set[str]) -> bool:
        """Return True if a settlement appendix entry was actually named in the dialogue.

        This intentionally does NOT keep entries merely because a ruler/governor/clan inside the
        entry was mentioned. Those broad matches caused specific settlements to survive just
        because the dialogue mentioned a broad title or family name. For economic_effects target_id,
        the useful signal is the settlement name/ID itself.
        """
        dialogue_l = dialogue_text.lower()
        name = self._settlement_entry_name(entry)
        if not name:
            return False
        if name in dialogue_l:
            return True
        normalized_name = re.sub(r"[^a-z0-9]+", " ", name).strip()
        normalized_dialogue = re.sub(r"[^a-z0-9]+", " ", dialogue_l)
        if normalized_name and normalized_name in normalized_dialogue:
            return True
        # Do not match by loose individual words. That caused false positives such as
        # "Flint's Finger" surviving because the dialogue contained unrelated words
        # "flint" and "finger" in different sentences. Exact normalized phrase matching
        # above is the safe behavior for this technical target-id appendix.
        return False

    def _filter_dialogue_settlement_appendix(self, appendix: str, dialogue_text: str) -> str:
        """Filter the giant 'Settlements mentioned in this dialogue' appendix to actually relevant entries."""
        lines = appendix.splitlines()
        if not lines:
            return ''
        header = lines[0]
        entries: List[List[str]] = []
        current: List[str] = []
        for line in lines[1:]:
            if re.match(r'^\s*-\s+', line):
                if current:
                    entries.append(current)
                current = [line]
            elif current:
                current.append(line)
        if current:
            entries.append(current)
        if not entries:
            return appendix.strip()

        tokens = self._dialogue_name_tokens(dialogue_text)
        kept: List[List[str]] = []
        for entry_lines in entries:
            entry_text = '\n'.join(entry_lines)
            if self._entry_matches_dialogue_text(entry_text, dialogue_text, tokens):
                kept.append(entry_lines)

        max_keep = max(self.max_event_dialogue_settlements, 0)
        if max_keep and len(kept) > max_keep:
            kept = kept[:max_keep]

        hidden = max(0, len(entries) - len(kept))
        if not kept:
            return f"{header}\n    (No settlement entries matched the actual dialogue; {len(entries)} entries hidden by proxy.)"
        rendered = [header] + ['\n'.join(e) for e in kept]
        if hidden:
            rendered.append(f"    ({hidden} other settlement entries hidden by proxy.)")
        return '\n'.join(rendered).strip()

    def _trim_event_dialogue_messages(self, dialogue_text: str) -> str:
        """Keep a bounded set of complete speaker messages inside a New dialogue block."""
        max_msgs = max(self.max_event_dialogue_messages, 0)
        if max_msgs <= 0:
            return dialogue_text.strip()
        lines = dialogue_text.splitlines()
        if not lines:
            return ''
        header = lines[0]
        body = lines[1:]
        msg_re = re.compile(r'^\s{2}(Player|[^:]{1,80}):')
        messages: List[List[str]] = []
        current: List[str] = []
        for line in body:
            if msg_re.match(line):
                if current:
                    messages.append(current)
                current = [line]
            elif current:
                current.append(line)
            elif line.strip():
                current = [line]
        if current:
            messages.append(current)
        if len(messages) <= max_msgs:
            return dialogue_text.strip()

        important_re = re.compile(r'\b(war|battle|siege|army|march|raid|burn|forage|grain|tribute|peace|alliance|ransom|prisoner|enemy|movement|commander|scout|banners)\b', re.IGNORECASE)
        selected_idx = set(range(max(0, len(messages) - max_msgs), len(messages)))
        for idx, msg in enumerate(messages):
            if important_re.search('\n'.join(msg)):
                selected_idx.add(idx)
        selected = [messages[i] for i in sorted(selected_idx)]
        hidden = len(messages) - len(selected)
        rendered = [header]
        if hidden > 0:
            rendered.append(f"  ({hidden} older dialogue messages hidden by proxy; key/world-impacting lines preserved.)")
        for msg in selected:
            rendered.extend(msg)
        return '\n'.join(rendered).strip()

    def _filter_event_dialogue_conversation_body(self, convo: str) -> str:
        """Trim a selected event/diplomacy conversation and filter giant technical appendices."""
        marker = '\n  **Settlements mentioned in this dialogue'
        pos = convo.find(marker)
        if pos == -1:
            return self._trim_event_dialogue_messages(convo)
        spoken = convo[:pos].rstrip()
        appendix_and_rest = convo[pos:].strip()
        # Stop settlement appendix before later markdown-ish subheaders if they exist.
        next_markers = ['\n  **Characters mentioned', '\n  **Parties mentioned', '\n### ', '\n=== ']
        end_rel = None
        for nm in next_markers:
            j = appendix_and_rest.find(nm, 1)
            if j != -1:
                end_rel = j if end_rel is None else min(end_rel, j)
        if end_rel is None:
            appendix = appendix_and_rest
            rest = ''
        else:
            appendix = appendix_and_rest[:end_rel].rstrip()
            rest = appendix_and_rest[end_rel:].lstrip()
        trimmed_spoken = self._trim_event_dialogue_messages(spoken)
        filtered_appendix = self._filter_dialogue_settlement_appendix(appendix, spoken)
        parts = [trimmed_spoken, filtered_appendix]
        if rest:
            parts.append(rest)
        return '\n'.join(p for p in parts if p.strip()).strip()

    def _filter_new_npc_dialogues(self, raw: str, context_content: str = '') -> str:
        """
        Filter the huge event prompt dialogue block by whole dialogue conversations.

        The event generator is told to ignore private/personal chatter and create significant
        public/world events. This filter mirrors that: keep whole conversations that contain
        event-worthy/public-world terms, plus remove exact duplicate conversations.
        """
        header_match = re.match(r'(?s)^(.*?NEW NPC DIALOGUES.*?\n)', raw)
        header = header_match.group(1).strip() if header_match else '=== NEW NPC DIALOGUES (since last analysis) ==='

        parts = re.split(r'\n(?=New dialogue with )', raw)
        conversations = [part.strip() for part in parts if part.strip().startswith('New dialogue with ')]
        if not conversations:
            return raw.strip()

        # Score only the dialogue itself, not the appended technical "mentioned settlements/characters"
        # blocks, otherwise every long conversation looks relevant just because it contains IDs.
        major_keywords = re.compile(
            r'\b('
            r'war|battle|siege|captur|conquer|army|host|march|raid|rebel|rebellion|peace|alliance|'
            r'treaty|tribute|demand|declare|surrender|ransom|prisoner|mercenar|vassal|kingdom|ruler|'
            r'marriage|betroth|murder|assassin|feud|scandal|spy|secret alliance|economic|grain|'
            r'blockade|famine|plague|trade|contract|serve the|service of|bannermen|levies'
            r')\b',
            re.IGNORECASE,
        )
        weak_context_keywords = re.compile(r'\b(lord|lady|clan|town|castle|village|settlement|string_id)\b', re.IGNORECASE)
        private_noise = re.compile(
            r'\b(flirt|beautiful|kiss|love|romance|bed|desire|horny|companions so far|how do you like|idle chatter)\b',
            re.IGNORECASE,
        )

        seen = set()
        scored: List[tuple[int, int, str]] = []
        for idx, convo in enumerate(conversations):
            dialogue_text = self._strip_dialogue_technical_appendices(convo)
            fingerprint = re.sub(r'\s+', ' ', dialogue_text).strip().lower()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)

            major_score = len(major_keywords.findall(dialogue_text))
            weak_score = len(weak_context_keywords.findall(dialogue_text))
            score = major_score * 3 + min(weak_score, 3)

            # Personal chatter should not become an event merely because it mentions a lord/lady.
            if private_noise.search(dialogue_text) and major_score < 2:
                score -= 4

            # Require at least one strong public/world-impacting signal.
            if major_score > 0 and score > 0:
                scored.append((score, idx, convo))

        # Keep selected conversations in original order. If nothing matched, keep no dialogues and let
        # the event generator return [] rather than sending 90k of private chatter.
        if not scored:
            return f"{header}\n\n(No public/world-impacting NPC dialogues selected by proxy filter.)"

        scored.sort(reverse=True, key=lambda item: (item[0], -item[1]))
        selected = scored[:max(self.MAX_EVENT_HISTORY, 1)]
        selected.sort(key=lambda item: item[1])
        filtered_convos = [self._filter_event_dialogue_conversation_body(item[2]) for item in selected]
        return '\n\n'.join([header] + [c for c in filtered_convos if c.strip()]).strip()


    def _extract_static_system_instructions(self, system_content: str) -> str:
        """
        Preserve preamble/static instructions while removing known bulk data sections.

        This is delimiter-based section replacement, not character-count truncation.
        """
        if not system_content.strip():
            return ''

        # Remove the whole incoming world.txt block before generic line-based stripping,
        # because world.txt itself contains internal markdown/equal-sign headers that should
        # not terminate the removal. The filtered version is re-added from the incoming block.
        system_content = self._strip_incoming_world_blocks_from_text(system_content)

        lines = system_content.splitlines()
        output_lines: List[str] = []
        skip_mode = False
        saw_recognized_section = False

        # Header styles seen in prompts/files:
        #   === WORLD LORE ===
        #   ## WORLD LORE
        #   WORLD LORE:
        #   world.txt:
        header_re = re.compile(
            r'^\s*(?:'
            r'={2,}\s*(.+?)\s*={2,}'
            r'|#{2,6}\s*(.+?)\s*'
            r'|([A-Za-z0-9_ .\-/]{2,80})\s*[:：]\s*'
            r')$'
        )

        for line in lines:
            if self._is_policy_end_marker_line(line):
                # Closing markers are parser controls only. They close the current authored
                # policy block and are not sent to the backend model. In static prompt cleanup,
                # this simply stops any active skip and lets following AIInfluence boilerplate
                # fall back to normal handling.
                skip_mode = False
                continue

            match = header_re.match(line)
            if match:
                title = next((g for g in match.groups() if g), '').strip()
                normalized = re.sub(r'[^a-z0-9]+', ' ', title.lower()).strip()
                underscored = normalized.replace(' ', '_')
                dotted = normalized.replace(' ', '.')
                if (
                    normalized in self.BULK_SECTION_ALIASES
                    or underscored in self.BULK_SECTION_ALIASES
                    or dotted in self.BULK_SECTION_ALIASES
                ):
                    skip_mode = True
                    saw_recognized_section = True
                    continue
                skip_mode = False

            if not skip_mode:
                output_lines.append(line)

        # If we saw recognized bulk sections, keep the remaining instructions.
        # If not, do not duplicate the whole unknown system prompt; the filtered prompt is rebuilt below.
        if saw_recognized_section:
            return '\n'.join(output_lines).strip()
        return ''

    def _build_system_prompt(
        self,
        original_system_content: str,
        filtered_content: Dict[str, str],
        request_type: str
    ) -> str:
        """Build reconstructed system prompt from filtered content."""
        parts = []

        static_instructions = self._extract_static_system_instructions(original_system_content)
        if static_instructions:
            parts.append(static_instructions)

        section_order = [
            'pinned_world_instructions',
            'rules_action_rules',
            'rules_statement_rules',
            'rules_event_rules',
            'rules_analyzer_rules',
            'rules_player_description',
            'world_lore',
            'cultural_traditions',
            'game_state',
            'ai_current_data',
            'ai_current_world_state',
            'ai_existing_events',
            'ai_active_economic_effects',
            'ai_recent_diplomatic_statements',
            'event_history',
            'dialogue_history',
            'ai_conversation_history',
            'ai_new_npc_dialogues',
        ]

        for section in section_order:
            if section in filtered_content:
                parts.append(f"\n\n{'=' * 40}\n{section.upper()}\n{'=' * 40}\n{filtered_content[section]}")

        for key, content in filtered_content.items():
            if key not in section_order:
                parts.append(f"\n\n{'=' * 40}\n{key.upper()}\n{'=' * 40}\n{content}")

        return '\n'.join(parts).strip()

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // self.CHARS_PER_TOKEN

    def get_filter_stats(self, filtered: FilteredPrompt) -> Dict[str, Any]:
        """Get statistics about filtering."""
        return {
            "original_chars": filtered.original_size,
            "filtered_chars": filtered.filtered_size,
            "original_tokens_est": filtered.original_size // self.CHARS_PER_TOKEN,
            "filtered_tokens_est": filtered.filtered_size // self.CHARS_PER_TOKEN,
            "reduction_percentage": filtered.reduction_pct,
            "sections_included": filtered.sections_included,
            "pinned_sections_included": filtered.pinned_sections_included,
            "entities_found": {
                k: len(v) for k, v in filtered.entities_found.items()
            },
        }
