# Copyright (C) 2026 SEGAREGA
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
GM Content Manager - Section-based selector filtering

Strategy:
1. Parse world.txt by section headers (===, ##)
2. Preserve structured sections and policy boundaries
3. Let the selector model decide which [GM] children to keep
4. Return FULL SECTIONS (not chunks)

For game state:
- Always include: wars, alliances, kingdom list
- GM selector filter: army details, economic effects

For rules files:
- Pass through intact (no filtering)
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from config.settings import Settings
from .static_index import StaticGMIndex

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """A section from world.txt or other structured content."""
    title: str
    full_content: str
    summary: str  # Generated summary for selector/context matching
    source: str
    level: int  # Header level (=== is 1, ## is 2)
    policy: str = "gm"  # "pinned", "gm", or "ignore"; inherited from parent headers when omitted
    explicit_policy: Optional[str] = None  # Original authored policy marker on this exact header, if any
    title_only_content: bool = False  # True for one-line policy entries such as [PINNED] text [END PIN]
    entities: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class SelectedSection:
    """A selected section with a selection reason."""
    section: Section
    reason: str  # Why it was selected


class GMContentManager:
    """
    Section-based GM content manager for world lore and game content.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.filtering_mode = settings.get_filtering_mode()

        # Section storage
        self.sections: List[Section] = []
        self.pinned_sections: List[Section] = []  # Always-included global world instructions
        
        # Entity index for fast lookup
        self.entity_index: Dict[str, List[int]] = {}  # entity -> section indices
        
        # Rules content (stored separately, not filtered)
        self.rules_content: Dict[str, str] = {}  # source -> full content
        
        # Cultural traditions (JSON, entity-matched)
        self.cultural_traditions: Dict[str, str] = {}  # culture_id -> tradition
        
        # File tracking
        self.file_mtimes: Dict[str, float] = {}

        self.max_event_history = settings.max_event_history
        self.static_gm_index: Optional[StaticGMIndex] = StaticGMIndex(settings) if getattr(settings, "static_gm_index_enabled", False) else None
    
    async def initialize(self):
        """Initialize prompt-only GM helpers.

        v9 intentionally does NOT read world.txt, rules files, cultural_traditions.json,
        or save-data files from disk. The proxy only filters the prompt-container message
        that AIInfluence actually sends to /v1/chat/completions.
        """
        logger.info(
            "Initializing prompt-only GM helper (mode=%s, static_index=%s)...",
            self.filtering_mode,
            bool(self.static_gm_index),
        )
        self.sections.clear()
        self.pinned_sections.clear()
        self.entity_index.clear()
        self.rules_content.clear()
        self.cultural_traditions.clear()
        self.file_mtimes.clear()
        if self.static_gm_index is not None:
            await self.static_gm_index.initialize()
    
    async def reindex(self):
        """Refresh the optional static [GM] index and clear legacy in-memory indexes."""
        logger.info("Prompt-only mode: clearing legacy indexes; refreshing static GM index when enabled")
        self.sections.clear()
        self.pinned_sections.clear()
        self.entity_index.clear()
        self.rules_content.clear()
        self.cultural_traditions.clear()
        self.file_mtimes.clear()
        if self.static_gm_index is not None:
            await self.static_gm_index.reindex()
        await self._save_cached_index()

    async def aclose(self):
        """Release optional static-index resources."""
        if self.static_gm_index is not None:
            await self.static_gm_index.aclose()
    

    POLICY_PINNED = "pinned"
    POLICY_GM = "gm"
    POLICY_IGNORE = "ignore"
    POLICY_TAG_RE = re.compile(r'^\s*\[(PINNED|PIN|GM|IGNORE)((?::[A-Z0-9_ -]+)*)\]\s*(.*?)\s*$', re.IGNORECASE)
    POLICY_END_TAG_RE = re.compile(r'^\s*\[END\s+(PINNED|PIN|GM|IGNORE)(?::[A-Z0-9_ -]+)*\]\s*\.?\s*$', re.IGNORECASE)
    INLINE_POLICY_END_RE = re.compile(r'\s*\[END\s+(PINNED|PIN|GM|IGNORE)(?::[A-Z0-9_ -]+)*\]\s*\.?\s*$', re.IGNORECASE)

    def _normalize_title(self, title: str) -> str:
        """Normalize section titles for matching."""
        # Remove machine policy tags before human/title matching.
        title = re.sub(r'\[(?:PINNED|PIN|GM|IGNORE|END\s+(?:PINNED|PIN|GM|IGNORE))(?::[A-Z0-9_ -]+)*\]', '', title, flags=re.IGNORECASE)
        return re.sub(r'[^a-z0-9]+', ' ', title.lower()).strip()

    def _collapse_duplicate_visible_wrappers(self, text: str) -> str:
        """Collapse wrapper artifacts without inventing a new style.

        This only repairs doubled wrappers produced by older builds, e.g.
        ``=== === Title === ===`` -> ``=== Title ===`` and
        ``### ### Title ###`` -> ``### Title ###``.
        """
        value = str(text or "").strip()
        for _ in range(3):
            previous = value
            value = re.sub(r'^(={2,})\s*((?:={2,})\s*.+?\s*(?:={2,}))\s*\1$', r'\2', value).strip()
            value = re.sub(r'^(#{1,6})\s+((?:#{1,6})\s+.+?)$', r'\2', value).strip()
            if value == previous:
                break
        return value

    def _strip_policy_control_markers_from_line(self, line: str) -> str:
        """Remove only policy-control markers from a visible line.

        The author controls every other character.  ``[GM] ### Title ###`` becomes
        ``### Title ###``; ``=== [PINNED] Title ===`` becomes ``=== Title ===``;
        ``[PINNED] body [END PINNED]`` becomes ``body``.
        """
        value = str(line or "").strip()
        if not value:
            return ""
        if self.POLICY_END_TAG_RE.match(value):
            return ""
        value = re.sub(
            r'\[(?:PINNED|PIN|GM|IGNORE)(?::[A-Z0-9_ -]+)*\]\s*',
            '',
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            r'\s*\[END\s+(?:PINNED|PIN|GM|IGNORE)(?::[A-Z0-9_ -]+)*\]\s*\.?',
            '',
            value,
            flags=re.IGNORECASE,
        )
        return self._collapse_duplicate_visible_wrappers(value).strip()

    def _visible_header_line(self, raw_line: str, clean_title: str) -> str:
        """Render authored text after removing only policy-control markers.

        The parser must not require or invent ``===``/``###`` title syntax.  Any
        line beginning with a policy marker is recognized, and only the marker is
        removed from the rendered prompt.
        """
        visible = self._strip_policy_control_markers_from_line(raw_line)
        if visible:
            return visible
        return self._collapse_duplicate_visible_wrappers(str(clean_title or "").strip())

    def _normalize_policy_name(self, policy: str) -> str:
        raw = str(policy or "").strip().lower()
        if raw == "pin":
            return self.POLICY_PINNED
        return raw

    def _strip_inline_policy_end(self, text: str) -> Tuple[str, Optional[str]]:
        """Remove a trailing inline close marker such as [END PIN] from title text."""
        value = str(text or "").strip()
        match = self.INLINE_POLICY_END_RE.search(value)
        if not match:
            return value, None
        closed_policy = self._normalize_policy_name(match.group(1))
        return value[:match.start()].strip(), closed_policy

    def _parse_policy_end_tag(self, title: str) -> Optional[str]:
        """Return the policy closed by an [END ...] marker, if this title is one."""
        match = self.POLICY_END_TAG_RE.match(title.strip())
        if not match:
            return None
        return self._normalize_policy_name(match.group(1))

    def _strip_policy_end_markers(self, text: str) -> str:
        """Remove standalone === [END PINNED/GM/IGNORE] === marker lines from text sent to the model."""
        return re.sub(
            r'(?im)^\s*(?:={3,}\s*)?\[END\s+(?:PINNED|PIN|GM|IGNORE)(?::[A-Z0-9_ -]+)*\](?:\s*={3,})?\s*\.?\s*\r?\n?',
            '',
            text,
        ).strip()

    def _request_type_aliases(self, request_type: Optional[str]) -> set:
        """Return normalized request-type aliases used by scoped policy tags.

        Examples:
            request_type="dialogue" -> {"dialogue"}
            request_type="event_generation" -> {"events", "event", "event_generation"}
        """
        raw = (request_type or "").lower().strip()
        norm = re.sub(r'[^a-z0-9]+', '_', raw).strip('_')
        aliases = {norm} if norm else set()
        if norm in {'chat', 'dialogue', 'dialog'}:
            # Treat legacy request_type='chat' as dialogue internally. Do not expose [*:CHAT] as a tag scope.
            aliases.update({'dialogue', 'dialog'})
        if 'event' in norm:
            aliases.update({'event', 'events', 'event_generation', 'eventgenerator', 'event_generator'})
        if 'diplom' in norm or 'statement' in norm:
            aliases.update({'diplomacy', 'diplomatic', 'statement', 'statements'})
        return {a for a in aliases if a}

    def _parse_policy_tag_details(self, title: str, request_type: Optional[str] = None) -> Tuple[Optional[str], str, bool]:
        """Extract a policy marker from the beginning of a header/title line.

        Supported examples:
            "[PINNED] General Lore"
            "[PIN] General Lore"
            "[GM:DIALOGUE] -- Cultures --"
            "[IGNORE] Draft notes"

        Everything after the marker is treated as the human title/content. The
        parser does not care whether the author uses dashes, asterisks, equals,
        or plain prose for that title. A trailing inline end marker such as
        "[END PIN]" is stripped and marks the one-line policy entry as closed.
        """
        match = self.POLICY_TAG_RE.match(title.strip())
        if not match:
            return None, title.strip(), False
        policy = self._normalize_policy_name(match.group(1))
        scope_blob = match.group(2) or ''
        clean_title = (match.group(3) or '').strip()
        clean_title, inline_closed_policy = self._strip_inline_policy_end(clean_title)
        inline_closed = inline_closed_policy is not None

        scopes = [s.strip().lower() for s in scope_blob.split(':') if s.strip()]
        if scopes:
            wanted = {re.sub(r'[^a-z0-9]+', '_', s).strip('_') for s in scopes}
            aliases = self._request_type_aliases(request_type)
            expanded = set(wanted)
            if 'event' in wanted:
                expanded.add('events')
            if 'events' in wanted:
                expanded.add('event')
            if 'chat' in wanted:
                expanded.add('dialogue')
            if 'diplomacy' in wanted:
                expanded.add('diplomatic')
            if aliases.isdisjoint(expanded):
                return self.POLICY_IGNORE, clean_title, inline_closed

        return policy, clean_title, inline_closed

    def _parse_policy_tag(self, title: str, request_type: Optional[str] = None) -> Tuple[Optional[str], str]:
        policy, clean_title, _inline_closed = self._parse_policy_tag_details(title, request_type=request_type)
        return policy, clean_title

    def _infer_legacy_world_policy(self, clean_title: str) -> str:
        """Fallback for untagged headers: keep them by default unless explicitly tagged otherwise."""
        return self.POLICY_PINNED

    def _is_empty_container_section(self, section: Section) -> bool:
        """True when a section has no visible body beyond its header line."""
        title = str(section.title or "").strip()
        if bool(getattr(section, "title_only_content", False)) and title:
            return False

        lines = section.full_content.splitlines()
        # Normal headed sections store their first line as the visible header.
        # Untitled preface sections, created from text before the first header, do not.
        # Do not skip their first line or a pinned preface such as
        # "- **The World:** [PINNED] ..." is mistaken for an empty container.
        start_index = 1 if title else 0
        for line in lines[start_index:]:
            stripped = line.strip()
            if stripped:
                return False
        # Some authored GM children intentionally store the entire content in the
        # header line itself, e.g.:
        #   ## In Dorne, women inherit equally...
        # Treat those as real content sections rather than empty containers.
        if section.level >= 2 and title:
            if len(title) >= 60 or len(title.split()) >= 8:
                return False
        return True

    async def _index_world_lore(self, path: str):
        """Disabled in prompt-only mode. The proxy never reads world.txt from disk."""
        logger.info("Prompt-only mode: skipping world_lore file ingestion")
        return

    def _parse_sections(self, content: str, source: str, request_type: Optional[str] = None) -> List[Section]:
        """
        Parse content into complete semantic sections based on headers.

        Supported headers:
            === [PINNED] Major Header ===
            == [GM] Sub Header ==
            ## Child Header
            ### Child Header

        Policy rules:
            - [PINNED] is always included.
            - [GM] is selected only when relevant.
            - [IGNORE] is never sent.
            - [PINNED:DIALOGUE], [GM:EVENTS:DIPLOMACY], etc. apply only to matching request types.
            - Untagged children inherit the nearest tagged parent policy.
            - Untagged top-level headers are kept by default unless explicitly marked [GM] or [IGNORE].

        A section's body continues until the next header of any level. This deliberately
        makes each `## Culture` entry a precise selection unit instead of allowing a huge
        parent section to swallow all children.
        """
        sections: List[Section] = []

        # Supported headers:
        #   === [GM] Title ===      legacy decorated policy header
        #   [GM] Title              compact policy header; everything after the marker is title
        #   ## Child Header         child candidate under the nearest policy parent
        header_pattern = re.compile(
            r'^(?:(={3,})\s*(.+?)\s*\1|(#{2,6})\s*(.+?)\s*|(\[(?:PINNED|PIN|GM|IGNORE|END\s+(?:PINNED|PIN|GM|IGNORE))(?::[A-Z0-9_ -]+)*\]\s*.*?))\s*$',
            re.IGNORECASE,
        )
        lines = content.split('\n')

        current_section_lines: List[str] = []
        current_title = ""
        current_level = 0
        current_policy = self.POLICY_PINNED
        current_explicit_policy: Optional[str] = None
        current_title_only_content = False
        policy_stack: Dict[int, str] = {}

        def save_current() -> None:
            if not current_section_lines or not any(str(line).strip() for line in current_section_lines):
                return
            full_content = '\n'.join(current_section_lines)
            sections.append(Section(
                title=current_title,
                full_content=full_content,
                summary="",
                source=source,
                level=current_level,
                policy=current_policy,
                explicit_policy=current_explicit_policy,
                title_only_content=current_title_only_content,
            ))

        for raw_line in lines:
            line = raw_line.strip()
            match = header_pattern.match(line)
            if not match:
                current_section_lines.append(raw_line)
                continue

            save_current()

            equals = match.group(1)
            equals_title = match.group(2)
            hashes = match.group(3)
            hash_title = match.group(4)
            bare_policy_title = match.group(5)

            if equals:
                level = 1 if len(equals) >= 3 else 2
                raw_title = (equals_title or '').strip()
            elif hashes:
                level = len(hashes or '#')
                raw_title = (hash_title or '').strip()
            else:
                level = 1
                raw_title = (bare_policy_title or '').strip()

            end_policy = self._parse_policy_end_tag(raw_title)
            if end_policy:
                # Close only the most recent open section/policy of that type. This is not
                # retroactive: previously saved sections are untouched. If a parent policy remains
                # (for example === [GM] Cultures === then ## [PINNED] Global Rule), following
                # unheaded text inherits that parent. Otherwise it falls back to normal keep/pinned.
                matching_levels = [lvl for lvl, pol in policy_stack.items() if pol == end_policy]
                if matching_levels:
                    del policy_stack[max(matching_levels)]
                inherited_after_end = None
                for parent_level in sorted(policy_stack.keys(), reverse=True):
                    inherited_after_end = policy_stack[parent_level]
                    break
                current_title = ""
                current_level = (max(policy_stack.keys()) + 1) if policy_stack else 0
                current_policy = inherited_after_end or self.POLICY_PINNED
                current_explicit_policy = None
                current_title_only_content = False
                current_section_lines = []
                continue

            explicit_policy, clean_title, inline_closed = self._parse_policy_tag_details(raw_title, request_type=request_type)

            # A child with no explicit tag inherits nearest parent policy.
            inherited_policy = None
            for parent_level in sorted(policy_stack.keys(), reverse=True):
                if parent_level < level:
                    inherited_policy = policy_stack[parent_level]
                    break

            effective_policy = explicit_policy or inherited_policy or self._infer_legacy_world_policy(clean_title)

            # Drop stale child policies and push the current header's effective policy so
            # descendants inherit it even if the policy was inferred or inherited. A one-line
            # inline-closed policy entry, e.g. [PINNED] text [END PIN], is saved immediately
            # and does not affect following lines.
            for stacked_level in list(policy_stack.keys()):
                if stacked_level >= level:
                    del policy_stack[stacked_level]
            if not inline_closed:
                policy_stack[level] = effective_policy

            current_title = clean_title
            current_level = level
            current_policy = effective_policy
            current_explicit_policy = explicit_policy
            current_title_only_content = bool(inline_closed and clean_title and not equals and not hashes)
            visible_header = self._visible_header_line(raw_line, clean_title)
            current_section_lines = [visible_header] if visible_header else []

            if inline_closed:
                save_current()
                inherited_after_end = None
                for parent_level in sorted(policy_stack.keys(), reverse=True):
                    inherited_after_end = policy_stack[parent_level]
                    break
                current_title = ""
                current_level = (max(policy_stack.keys()) + 1) if policy_stack else 0
                current_policy = inherited_after_end or self.POLICY_PINNED
                current_explicit_policy = None
                current_title_only_content = False
                current_section_lines = []

        save_current()
        return sections

    def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract entities from the prompt text itself using generic structured labels."""
        entities = {
            'kingdoms': [],
            'characters': [],
            'locations': [],
            'cultures': [],
            'string_ids': []
        }

        labeled_fields = {
            'kingdoms': (
                'kingdom',
                'faction',
                'realm',
                'house',
                'clan',
                'side',
                'party faction',
            ),
            'characters': (
                'leader',
                'ruler',
                'governor',
                'notable',
                'speaker',
                'npc',
                'character',
                'companion',
                'lord',
                'lady',
                'king',
                'queen',
                'prince',
                'princess',
                'emperor',
                'empress',
            ),
            'locations': (
                'location',
                'current location',
                'settlement',
                'town',
                'village',
                'castle',
                'city',
                'region',
            ),
            'cultures': (
                'culture',
                'character culture',
                'player culture',
                'religion',
                'faith',
                'belief',
            ),
        }

        for match in re.finditer(r'\b(?:string_id|id)\s*:\s*"?(?P<id>[A-Za-z0-9_:-]+)"?', content, re.IGNORECASE):
            cleaned = self._clean_entity_value(match.group('id'))
            if cleaned:
                entities['string_ids'].append(cleaned)

        for bucket, labels in labeled_fields.items():
            pattern = re.compile(
                r'(?im)^\s*(?:' + '|'.join(re.escape(label) for label in labels) + r')\s*:\s*(?P<value>.+?)\s*$'
            )
            for match in pattern.finditer(content):
                for raw_value in self._split_entity_values(match.group('value')):
                    cleaned = self._clean_entity_value(raw_value)
                    if cleaned:
                        entities[bucket].append(cleaned)

        for key in entities:
            seen = set()
            deduped = []
            for e in entities[key]:
                if not e or len(e) <= 2:
                    continue
                norm = str(e).lower()
                if norm in seen:
                    continue
                seen.add(norm)
                deduped.append(e)
            entities[key] = deduped[:10]

        return entities

    def _split_entity_values(self, value: str) -> List[str]:
        """Split a structured field value into candidate entity phrases."""
        text = str(value or '').strip()
        if not text:
            return []
        parts = re.split(r'\s*(?:;|\||/|\band\b)\s*', text, flags=re.IGNORECASE)
        return [part for part in parts if part.strip()]

    def _clean_entity_value(self, value: str) -> str:
        """Normalize a raw structured field value into a stable entity phrase."""
        text = str(value or '').strip().strip("\"'`*- ")
        if not text:
            return ''
        text = re.split(r'\s+\((?:id|string_id|relations?|culture|faction|clan|age)\s*:', text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = re.split(r'\s+\|\s+', text, maxsplit=1)[0]
        text = re.split(r'\s+-\s+', text, maxsplit=1)[0]
        text = re.sub(r'\s+', ' ', text).strip(' ,;:.')
        if len(text) < 3:
            return ''
        if text.lower() in {'none', 'unknown', 'n/a', 'null'}:
            return ''
        return text
    
    def _generate_summary(self, section: Section) -> str:
        """
        Generate a concise summary for selector/context matching.
        This is the selector summary used when no static DB summary is available.
        """
        parts = [section.title]

        # Add key entities
        if section.entities.get('kingdoms'):
            parts.append(f"Kingdoms: {', '.join(section.entities['kingdoms'][:5])}")
        if section.entities.get('characters'):
            parts.append(f"Characters: {', '.join(section.entities['characters'][:3])}")
        if section.entities.get('cultures'):
            parts.append(f"Cultures: {', '.join(section.entities['cultures'][:5])}")
        if section.entities.get('locations'):
            parts.append(f"Locations: {', '.join(section.entities['locations'][:5])}")
        if section.entities.get('string_ids'):
            parts.append(f"IDs: {', '.join(section.entities['string_ids'][:5])}")

        # Include several meaningful body lines, not just the first one. This keeps later signals
        # like "Folklore: ... Faceless Men" available to the selector without hardcoding
        # any world-specific concepts.
        body_lines: List[str] = []
        for raw_line in section.full_content.split('\n')[1:]:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('==') or line.startswith('#'):
                continue
            body_lines.append(re.sub(r'\s+', ' ', line))

        selected_lines: List[str] = []
        char_budget = 700
        used = 0

        labeled_lines = [
            line for line in body_lines
            if re.match(r'^[A-Za-z][A-Za-z0-9 /\'’"(),._-]{0,50}\s*:\s*.+$', line)
        ]
        fallback_lines = [line for line in body_lines if line not in labeled_lines]

        prioritized_lines: List[str] = []
        for line in labeled_lines[:3]:
            if line not in prioritized_lines:
                prioritized_lines.append(line)
        for line in labeled_lines[-3:]:
            if line not in prioritized_lines:
                prioritized_lines.append(line)
        for line in labeled_lines[3:-3]:
            if line not in prioritized_lines:
                prioritized_lines.append(line)
        for line in fallback_lines:
            if line not in prioritized_lines:
                prioritized_lines.append(line)

        for line in prioritized_lines:
            clipped = line[:120].strip()
            if not clipped:
                continue
            projected = used + len(clipped) + (1 if selected_lines else 0)
            if projected > char_budget and selected_lines:
                break
            selected_lines.append(clipped)
            used = projected
            if len(selected_lines) >= 8:
                break

        if selected_lines:
            parts.extend(selected_lines)

        summary = ' '.join(parts)

        # Keep summaries reasonably compact while preserving multiple selector signals.
        if len(summary) > 900:
            summary = summary[:900]

        return summary
    
    async def _index_rules_file(self, path: str, name: str):
        """Disabled in prompt-only mode. The proxy never reads rule files from disk."""
        logger.info("Prompt-only mode: skipping rules file ingestion for %s", name)
        return

    async def _index_cultural_traditions(self, path: str):
        """Disabled in prompt-only mode. The proxy never reads cultural_traditions.json."""
        logger.info("Prompt-only mode: skipping cultural traditions file ingestion")
        return

    async def retrieve(
        self,
        query: str,
        entities: Dict[str, List[str]],
        request_type: str,
    ) -> Dict[str, Any]:
        """
        Collect relevant prompt content for a query.
        
        Returns:
            {
                'world_sections': List[SelectedSection],  # Full sections from world.txt
                'rules_content': Dict[str, str],           # All rules files intact
                'cultural_traditions': Dict[str, str],     # Relevant traditions only
                'always_include': Dict[str, Any]           # Wars, alliances, kingdom list
            }
        """
        result = {
            'world_sections': [],
            'rules_content': self.rules_content.copy(),  # ALL rules intact
            'cultural_traditions': {},
            'always_include': {},
            'pinned_world_sections': list(self.pinned_sections),
        }
        
        # Filter cultural traditions by exact prompt-derived matches only.
        normalized_traditions = {
            self._normalize_title(key): key
            for key in self.cultural_traditions
            if self._normalize_title(key)
        }
        for bucket in ('kingdoms', 'cultures', 'locations'):
            for value in entities.get(bucket, []):
                norm = self._normalize_title(str(value))
                original_key = normalized_traditions.get(norm)
                if original_key:
                    result['cultural_traditions'][original_key] = self.cultural_traditions[original_key]

        for string_id in entities.get('string_ids', []):
            string_id = str(string_id)
            if string_id in self.cultural_traditions:
                result['cultural_traditions'][string_id] = self.cultural_traditions[string_id]
        
        logger.debug(f"Filtered to {len(result['cultural_traditions'])} relevant traditions")
        
        return result

    async def select_relevant_sections(
        self,
        sections: List[Section],
        query: str,
        entities: Dict[str, List[str]],
    ) -> List[SelectedSection]:
        """Select relevant already-parsed [GM] sections using the static GM index."""
        if self.static_gm_index is not None:
            ranked_sections = self.static_gm_index.rank_prompt_sections(
                sections=sections,
                query=query,
                entities=entities,
                request_type="",
            )
            return [
                SelectedSection(
                    section=section,
                    reason="all_gm_candidates",
                )
                for section in ranked_sections
            ]

        logger.warning("Static GM index is disabled; keeping all candidate sections")
        return [
            SelectedSection(section=section, reason="static_index_unavailable")
            for section in sections
            if not self._is_empty_container_section(section)
        ]

    def get_rules_content(self) -> Dict[str, str]:
        """
        Get all rules content (never filtered).
        """
        return self.rules_content.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        return {
            "filtering_mode": self.filtering_mode,
            "static_gm_index": self.static_gm_index.get_stats() if self.static_gm_index else {"enabled": False},
            "total_sections": len(self.sections),
            "pinned_world_sections": len(self.pinned_sections),
            "selectable_world_sections": len(self.sections),
            "unique_entities": len(self.entity_index),
            "rules_files": len(self.rules_content),
            "cultural_traditions": len(self.cultural_traditions),
            "sources": list(set(s.source for s in self.sections)),
        }
    
    async def _save_cached_index(self):
        """Save index cache for faster startup."""
        # TODO: Implement disk caching
        pass
    
    async def _load_cached_index(self):
        """Load from cache if available."""
        await self.reindex()  # Fallback to full reindex
    
    def clear_cache(self):
        """No-op retained for compatibility."""
        logger.info("clear_cache called in selector-only mode; nothing to clear")


    def _normalize_query_text(self, query: str) -> str:
        text = re.sub(r'\s+', ' ', str(query or '')).strip()
        return text or "general context"

    def _normalized_entity_values(self, entities: Dict[str, List[str]]) -> List[str]:
        values: List[str] = []
        for bucket in ("kingdoms", "characters", "locations", "cultures", "string_ids"):
            for value in entities.get(bucket, []):
                normalized = self._normalize_title(str(value))
                if normalized and normalized not in values:
                    values.append(normalized)
        return values

    def _relevance_boost(self, section: Section, normalized_query: str, entity_values: List[str]) -> Tuple[float, str]:
        search_text = self._normalize_title(
            " ".join(
                part for part in (
                    section.title,
                    section.summary,
                    "\n".join(section.full_content.splitlines()[1:4]),
                ) if part
            )
        )
        if not search_text:
            return 0.0, ""

        query_tokens = [token for token in normalized_query.split() if len(token) >= 4]
        overlap = sum(1 for token in query_tokens if token in search_text)
        lexical_boost = min(0.08, overlap * 0.015)

        entity_hits = sum(1 for value in entity_values if value and value in search_text)
        entity_boost = min(0.12, entity_hits * 0.04)

        boost = lexical_boost + entity_boost
        reason_bits: List[str] = []
        if entity_hits:
            reason_bits.append(f"entity_hits={entity_hits}")
        if overlap:
            reason_bits.append(f"token_overlap={overlap}")
        return boost, ",".join(reason_bits)
