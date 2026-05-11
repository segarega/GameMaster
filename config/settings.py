# Copyright (C) 2026 SEGAREGA
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
GameMaster - Configuration Settings
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


REQUEST_TYPES = ("dialogue", "diplomacy", "events")
REQUEST_PARAMETER_NAMES = ("temperature", "top_p", "top_k")


DEFAULT_SELECTOR_INSTRUCTION = (
    'You are a proxy-side selector. '
    'Use the provided request context and candidate indexed entries to decide which candidate entry IDs should be kept. '
    'Resolve implication, pronouns, vague references, allegiance, location, recent dialogue, speaker identity, and relevant social context. '
    'Select from the provided candidates only. Never rewrite, summarize, or invent content. '
    'When uncertain, prefer keeping slightly too much over omitting needed context. '
    'Return strict JSON only in this shape: {"blocks":[{"block_id":"block_1","keep_ids":["id1","id2"]}]}.'
)

DEFAULT_FILTERING_MODE = "selector"


DEFAULT_SUMMARY_INSTRUCTION = (
    'You are summarizing one indexed text entry so another AI can later decide whether this entry is relevant to a request. '
    'Return only one concise paragraph, ideally 45-80 words. '
    "Preserve the entry's core meaning, important details, constraints, context, terms that may be useful for matching, "
    'and any details that would change when the full entry should or should not be included. '
    'Do not invent, generalize beyond the text, or add outside knowledge. Do not mention that this is a summary. '
    'Do not use markdown, bullets, headings, or labels unless they are necessary to preserve meaning. '
    'Compress aggressively, but keep enough concrete detail that the later AI can reliably decide whether the full entry should be included.'
)
DEFAULT_STATIC_GM_INDEX_FILES = [
    "world.txt",
    "actionrules.txt",
    "battlecombatrules.txt",
    "eventsanalyzerrules.txt",
    "eventsgeneratorrules.txt",
    "kingdomstatementrules.txt",
]


DEFAULT_CHARACTER_MEMORY_SUMMARY_PROMPT = (
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

DEFAULT_CHARACTER_MEMORY_PROFILE_PROMPT = (
    "You are conservatively updating a game character profile using conversation history. "
    "You may update the character's personality or backstory only when the conversation reveals durable, meaningful "
    "information that should affect future roleplay. Examples include new relationships, loyalties, grudges, debts, "
    "promises, losses, family news, imprisonment, release, betrayal, alliance, fear, respect, or changed opinion of the "
    "player. Do not rewrite the character into a different person. Preserve their established temperament, social status, "
    "history, culture, speech style, values, and contradictions unless the conversation gives strong evidence of gradual "
    "change. Do not turn a cruel character kind, a cynical character trusting, a noble-born character lowborn, or a "
    "lifelong enemy into a friend without strong evidence. Prefer small additive edits over broad rewrites. Keep the "
    "existing structure and style where possible. If no meaningful durable change is needed, return changed=false. "
    'Return strict JSON only with this shape: {"changed": true or false, "new_personality": string or null, '
    '"new_backstory": string or null, "reason": string, "confidence": "low" | "medium" | "high"}.'
)


def default_static_gm_index_files() -> List[str]:
    return list(DEFAULT_STATIC_GM_INDEX_FILES)


def default_selector_context_rules() -> List[Dict[str, Any]]:
    """Default marker-based extracts sent to the selector model."""
    return [
        {
            "name": "Character Briefing Context",
            "request_types": ["dialogue"],
            "beginning": "### Character Briefing (CURRENT DATA) ###",
            "end": "**Description:**",
            "include_beginning_marker": True,
            "include_end_marker": False,
        },
        {
            "name": "Conversation History Context",
            "request_types": ["dialogue"],
            "beginning": "### Conversation History ###",
            "end": "Last Interaction:",
            "include_beginning_marker": True,
            "include_end_marker": False,
        },
    ]


def default_request_parameters() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Defaults are disabled so unchecked parameters are not sent upstream."""
    return {
        request_type: {
            "temperature": {"enabled": False, "value": 1.0},
            "top_p": {"enabled": False, "value": 1.0},
            "top_k": {"enabled": False, "value": 40},
        }
        for request_type in REQUEST_TYPES
    }


def default_system_prompts() -> Dict[str, Dict[str, str]]:
    """Optional per-request-type system prompts inserted before/after the intercepted history."""
    return {
        request_type: {"pre_history": "", "post_history": ""}
        for request_type in REQUEST_TYPES
    }


def _bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _coerce_number(parameter_name: str, value: Any) -> float | int | None:
    if value is None or value == "":
        return None
    try:
        if parameter_name == "top_k":
            return int(value)
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class Settings:
    """
    Flat runtime configuration for GameMaster.

    Existing settings.json files are preserved. The "gm" section stores dynamic prompt filtering settings.
    """

    # Server
    host: str = "localhost"
    port: int = 5100

    # LLM Backend
    # Accepts either a base URL like https://openrouter.ai/api/v1
    # or a full /chat/completions URL like https://openrouter.ai/api/v1/chat/completions.
    api_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    models: Dict[str, str] = field(default_factory=lambda: {
        "events": "gpt-4-turbo",
        "diplomacy": "gpt-4-turbo",
        "dialogue": "gpt-4-turbo",
    })
    site_url: str = ""      # Optional OpenRouter HTTP-Referer
    app_title: str = ""     # Optional OpenRouter X-OpenRouter-Title

    # Request type detection. These signatures are checked against the intercepted prompt text.
    request_type_signatures: Dict[str, List[str]] = field(default_factory=lambda: {
        "dialogue": [
            "### Mission ###\nRole-play as a character in Mount & Blade II: Bannerlord. Use your personality, history, and context to inform responses. Output ONLY a valid JSON object with no extra text or markdown."
        ],
        "events": [
            "## EVENT STRUCTURE:\nMUST include: 1) CAUSE (from data) 2) ACTION (decision taken) 3) CONSEQUENCE (future impact)\nPrefer DEVELOPING existing conflicts over new minor incidents. Return [] if insufficient data."
        ],
        "diplomacy": [
            "### CRITICAL REMINDER: You Are a Living Ruler ###"
        ],
    })

    # GM filtering
    max_event_history: int = 20
    dialogue_history_size: int = 5
    dynamic_filter_enabled: bool = True
    fuzzy_match_threshold: float = 0.88
    max_people_present: int = 8
    max_nearby_settlements: int = 8
    max_nearby_parties: int = 8
    max_inventory_lines: int = 8
    max_event_dialogue_messages: int = 14
    max_event_dialogue_settlements: int = 10
    prompt_drop_rules: List[Dict[str, Any]] = field(default_factory=list)
    prompt_replace_rules: List[Dict[str, Any]] = field(default_factory=list)

    # Policy-header filtering backend.
    filtering_mode: str = DEFAULT_FILTERING_MODE

    # Per-request-type upstream sampling controls. Disabled entries are omitted from outbound JSON.
    request_parameters: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=default_request_parameters)

    # Optional per-request-type system prompts inserted as first and last system prompts.
    system_prompts: Dict[str, Dict[str, str]] = field(default_factory=default_system_prompts)


    # Optional raw upstream LLM request/response logging.
    llm_log_enabled: bool = False
    llm_log_path: str = "logs/llm_requests_responses.txt"
    llm_log_pretty_json: bool = True

    # Optional proxy-side selector model for [GM] sections.
    selector_enabled: bool = True
    selector_api_url: str = ""
    selector_api_key: str = ""
    selector_model: str = ""
    selector_temperature: float = 0.0
    selector_max_tokens: int = 1200
    selector_timeout_seconds: float = 120.0
    selector_instruction: str = DEFAULT_SELECTOR_INSTRUCTION
    selector_context_rules: List[Dict[str, Any]] = field(default_factory=default_selector_context_rules)
    selector_log_enabled: bool = False
    selector_log_path: str = "logs/selector-log.txt"
    selector_log_pretty_json: bool = True

    # Static [GM] index. Editable AIInfluence
    # files are indexed into a local DB, summarized once, then the selector sees only
    # summaries/IDs for all [GM] child elements.
    static_gm_index_enabled: bool = True
    static_gm_index_ai_influence_folder: str = ""
    static_gm_index_files: List[str] = field(default_factory=default_static_gm_index_files)
    static_gm_index_db_path: str = "cache/static_gm_index.sqlite3"
    # Deprecated compatibility flag. Static GM Index reindexing is manual-only.
    static_gm_index_auto_reindex: bool = False
    static_gm_index_selector_payload: str = "summary"
    static_gm_index_summary_enabled: bool = True
    static_gm_index_summary_api_url: str = ""
    static_gm_index_summary_api_key: str = ""
    static_gm_index_summary_model: str = ""
    static_gm_index_summary_temperature: float = 0.1
    static_gm_index_summary_max_tokens: int = 220
    static_gm_index_summary_timeout_seconds: float = 120.0
    static_gm_index_summary_max_chars: int = 6000
    static_gm_index_summary_instruction: str = DEFAULT_SUMMARY_INSTRUCTION

    # Character Memory Control. Edits AIInfluence save_data campaign JSONs so
    # ConversationHistory stays compact before AIInfluence builds prompts.
    character_memory_enabled: bool = True
    character_memory_campaign_dir: str = ""
    character_memory_api_url: str = ""
    character_memory_api_key: str = ""
    character_memory_model: str = ""
    character_memory_temperature: float = 0.1
    character_memory_max_tokens: int = 700
    character_memory_timeout_seconds: float = 180.0
    character_memory_preserve_last_lines: int = 10
    character_memory_auto_enabled: bool = False
    character_memory_auto_trigger_raw_lines: int = 16
    character_memory_auto_scan_interval_seconds: float = 30.0
    character_memory_auto_debounce_seconds: float = 8.0
    character_memory_max_memory_entries: int = 5
    character_memory_summary_prompt: str = DEFAULT_CHARACTER_MEMORY_SUMMARY_PROMPT
    character_memory_profile_update_prompt: str = DEFAULT_CHARACTER_MEMORY_PROFILE_PROMPT

    # GUI-only settings.
    gui_log_viewer_enabled: bool = False

    config_path: Optional[str] = None

    @classmethod
    def load(cls, config_path: Optional[str] = None, *, apply_env_overrides: bool = True) -> 'Settings':
        """Load settings from JSON file, optionally applying environment overrides."""
        if config_path is None:
            env_config_path = os.environ.get("GMR_CONFIG_PATH")
            if env_config_path and Path(env_config_path).exists():
                config_path = env_config_path
            else:
                search_paths = [
                    Path("config/settings.json"),
                    Path("settings.json"),
                    Path.home() / ".gamemaster_gm" / "settings.json",
                    Path.home() / ".gamemaster_raag" / "settings.json",
                ]

                for loc in search_paths:
                    if loc.exists():
                        config_path = str(loc)
                        break

        settings = cls()

        if config_path and Path(config_path).exists():
            settings._load_from_json(config_path)

        if apply_env_overrides:
            settings._apply_env_overrides()
        return settings

    def _normalize_request_parameters(self, raw: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
        normalized = default_request_parameters()
        if not isinstance(raw, dict):
            return normalized

        for request_type in REQUEST_TYPES:
            rt_data = raw.get(request_type, {})
            if not isinstance(rt_data, dict):
                continue
            for parameter_name in REQUEST_PARAMETER_NAMES:
                spec = rt_data.get(parameter_name, {})
                if isinstance(spec, dict):
                    enabled = _bool_from_any(spec.get("enabled", False))
                    value = spec.get("value", normalized[request_type][parameter_name]["value"])
                else:
                    # Legacy compact form: "temperature": 0.7 means enabled.
                    enabled = spec is not None
                    value = spec
                coerced = _coerce_number(parameter_name, value)
                normalized[request_type][parameter_name] = {
                    "enabled": enabled,
                    "value": coerced if coerced is not None else normalized[request_type][parameter_name]["value"],
                }
        return normalized

    def _normalize_system_prompts(self, raw: Any) -> Dict[str, Dict[str, str]]:
        normalized = default_system_prompts()
        if not isinstance(raw, dict):
            return normalized
        for request_type in REQUEST_TYPES:
            rt_data = raw.get(request_type, {})
            if isinstance(rt_data, dict):
                normalized[request_type]["pre_history"] = str(rt_data.get("pre_history", "") or "")
                normalized[request_type]["post_history"] = str(rt_data.get("post_history", "") or "")
        return normalized

    def _load_from_json(self, path: str):
        """Load settings from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Server
            if 'server' in data:
                self.host = data['server'].get('host', self.host)
                self.port = int(data['server'].get('port', self.port))

            # LLM
            if 'llm' in data:
                llm = data['llm']
                self.api_url = llm.get('api_url', self.api_url)
                self.api_key = llm.get('api_key', self.api_key)
                self.site_url = llm.get('site_url', self.site_url)
                self.app_title = llm.get('app_title', self.app_title)
                if 'models' in llm and isinstance(llm['models'], dict):
                    self.models.update(llm['models'])

            # Request type detection signatures.
            if 'request_type_detection' in data and isinstance(data['request_type_detection'], dict):
                detection = data['request_type_detection']
                for key in REQUEST_TYPES:
                    val = detection.get(key)
                    if isinstance(val, list):
                        self.request_type_signatures[key] = [str(x) for x in val if str(x)]
                    elif isinstance(val, str) and val:
                        self.request_type_signatures[key] = [val]

            # GM filtering.
            gm_data = data.get('gm') if isinstance(data.get('gm'), dict) else None

            if gm_data:
                gm = gm_data
                self.max_event_history = int(gm.get('max_event_history', self.max_event_history))
                self.dialogue_history_size = int(gm.get('dialogue_history_size', self.dialogue_history_size))
                self.dynamic_filter_enabled = bool(gm.get('dynamic_filter_enabled', self.dynamic_filter_enabled))
                self.fuzzy_match_threshold = float(gm.get('fuzzy_match_threshold', self.fuzzy_match_threshold))
                self.max_people_present = int(gm.get('max_people_present', self.max_people_present))
                self.max_nearby_settlements = int(gm.get('max_nearby_settlements', self.max_nearby_settlements))
                self.max_nearby_parties = int(gm.get('max_nearby_parties', self.max_nearby_parties))
                self.max_inventory_lines = int(gm.get('max_inventory_lines', self.max_inventory_lines))
                self.max_event_dialogue_messages = int(gm.get('max_event_dialogue_messages', self.max_event_dialogue_messages))
                self.max_event_dialogue_settlements = int(gm.get('max_event_dialogue_settlements', self.max_event_dialogue_settlements))

            self.filtering_mode = DEFAULT_FILTERING_MODE

            if 'static_gm_index' in data and isinstance(data['static_gm_index'], dict):
                idx = data['static_gm_index']
                self.static_gm_index_enabled = _bool_from_any(idx.get('enabled', self.static_gm_index_enabled))
                self.static_gm_index_ai_influence_folder = str(idx.get('ai_influence_folder', self.static_gm_index_ai_influence_folder) or '')
                if isinstance(idx.get('files'), list):
                    self.static_gm_index_files = [str(x) for x in idx.get('files', []) if str(x).strip()] or default_static_gm_index_files()
                self.static_gm_index_db_path = str(idx.get('db_path', self.static_gm_index_db_path) or self.static_gm_index_db_path)
                self.static_gm_index_auto_reindex = False  # manual-only; ignore old auto_reindex config
                self.static_gm_index_selector_payload = str(idx.get('selector_payload', self.static_gm_index_selector_payload) or 'summary').strip().lower()
                self.static_gm_index_summary_enabled = _bool_from_any(idx.get('summary_enabled', self.static_gm_index_summary_enabled))
                self.static_gm_index_summary_api_url = str(idx.get('summary_api_url', self.static_gm_index_summary_api_url) or '')
                self.static_gm_index_summary_api_key = str(idx.get('summary_api_key', self.static_gm_index_summary_api_key) or '')
                self.static_gm_index_summary_model = str(idx.get('summary_model', self.static_gm_index_summary_model) or '')
                self.static_gm_index_summary_temperature = float(idx.get('summary_temperature', self.static_gm_index_summary_temperature))
                self.static_gm_index_summary_max_tokens = int(idx.get('summary_max_tokens', self.static_gm_index_summary_max_tokens))
                self.static_gm_index_summary_timeout_seconds = float(idx.get('summary_timeout_seconds', self.static_gm_index_summary_timeout_seconds))
                self.static_gm_index_summary_max_chars = int(idx.get('summary_max_chars', self.static_gm_index_summary_max_chars))
                self.static_gm_index_summary_instruction = str(idx.get('summary_instruction', self.static_gm_index_summary_instruction) or self.static_gm_index_summary_instruction)

            if 'character_memory' in data and isinstance(data['character_memory'], dict):
                cm = data['character_memory']
                self.character_memory_enabled = _bool_from_any(cm.get('enabled', self.character_memory_enabled))
                self.character_memory_campaign_dir = str(cm.get('campaign_dir', self.character_memory_campaign_dir) or '')
                self.character_memory_api_url = str(cm.get('api_url', self.character_memory_api_url) or '')
                self.character_memory_api_key = str(cm.get('api_key', self.character_memory_api_key) or '')
                self.character_memory_model = str(cm.get('model', self.character_memory_model) or '')
                self.character_memory_temperature = float(cm.get('temperature', self.character_memory_temperature))
                self.character_memory_max_tokens = int(cm.get('max_tokens', self.character_memory_max_tokens))
                self.character_memory_timeout_seconds = float(cm.get('timeout_seconds', self.character_memory_timeout_seconds))
                self.character_memory_preserve_last_lines = int(cm.get('preserve_last_lines', self.character_memory_preserve_last_lines))
                self.character_memory_auto_enabled = _bool_from_any(cm.get('auto_enabled', self.character_memory_auto_enabled))
                self.character_memory_auto_trigger_raw_lines = int(cm.get('auto_trigger_raw_lines', self.character_memory_auto_trigger_raw_lines))
                self.character_memory_auto_scan_interval_seconds = float(cm.get('auto_scan_interval_seconds', self.character_memory_auto_scan_interval_seconds))
                self.character_memory_auto_debounce_seconds = float(cm.get('auto_debounce_seconds', self.character_memory_auto_debounce_seconds))
                self.character_memory_max_memory_entries = int(cm.get('max_memory_entries', self.character_memory_max_memory_entries))
                self.character_memory_summary_prompt = str(cm.get('summary_prompt', self.character_memory_summary_prompt) or self.character_memory_summary_prompt)
                self.character_memory_profile_update_prompt = str(cm.get('profile_update_prompt', self.character_memory_profile_update_prompt) or self.character_memory_profile_update_prompt)

            # Top-level prompt_drop_rules / prompt_replace_rules are also supported for readability.
            if 'prompt_drop_rules' in data and isinstance(data['prompt_drop_rules'], list):
                self.prompt_drop_rules = data['prompt_drop_rules']
            if 'prompt_replace_rules' in data and isinstance(data['prompt_replace_rules'], list):
                self.prompt_replace_rules = data['prompt_replace_rules']

            if 'request_parameters' in data:
                self.request_parameters = self._normalize_request_parameters(data.get('request_parameters'))

            if 'system_prompts' in data:
                self.system_prompts = self._normalize_system_prompts(data.get('system_prompts'))


            # Optional raw upstream request/response logging.
            if 'llm_logging' in data and isinstance(data['llm_logging'], dict):
                llm_logging = data['llm_logging']
                self.llm_log_enabled = bool(llm_logging.get('enabled', self.llm_log_enabled))
                self.llm_log_path = llm_logging.get('path', self.llm_log_path)
                self.llm_log_pretty_json = bool(llm_logging.get('pretty_json', self.llm_log_pretty_json))

            if 'selector' in data and isinstance(data['selector'], dict):
                selector = data['selector']
                self.selector_enabled = bool(selector.get('enabled', self.selector_enabled))
                self.selector_api_url = selector.get('api_url', self.selector_api_url)
                self.selector_api_key = selector.get('api_key', self.selector_api_key)
                self.selector_model = selector.get('model', self.selector_model)
                self.selector_temperature = float(selector.get('temperature', self.selector_temperature))
                self.selector_max_tokens = int(selector.get('max_tokens', self.selector_max_tokens))
                self.selector_timeout_seconds = float(selector.get('timeout_seconds', self.selector_timeout_seconds))
                self.selector_instruction = selector.get('instruction', self.selector_instruction)
                if 'context_rules' in selector and isinstance(selector['context_rules'], list):
                    self.selector_context_rules = selector['context_rules']
                self.selector_log_enabled = bool(selector.get('log_enabled', self.selector_log_enabled))
                self.selector_log_path = selector.get('log_path', self.selector_log_path)
                self.selector_log_pretty_json = bool(selector.get('pretty_json', self.selector_log_pretty_json))

            if 'gui' in data and isinstance(data['gui'], dict):
                self.gui_log_viewer_enabled = bool(data['gui'].get('log_viewer_enabled', self.gui_log_viewer_enabled))

            self.config_path = path

        except Exception as e:
            print(f"Error loading settings from {path}: {e}")

    def _apply_env_overrides(self):
        """Apply environment variable overrides (GMR_<KEY>)."""
        env_mapping = {
            'GMR_HOST': ('host', str),
            'GMR_PORT': ('port', int),
            'GMR_API_URL': ('api_url', str),
            'GMR_API_KEY': ('api_key', str),
            'GMR_SITE_URL': ('site_url', str),
            'GMR_APP_TITLE': ('app_title', str),
            'GMR_CHARACTER_MEMORY_CAMPAIGN_DIR': ('character_memory_campaign_dir', str),
            'GMR_CHARACTER_MEMORY_API_URL': ('character_memory_api_url', str),
            'GMR_CHARACTER_MEMORY_API_KEY': ('character_memory_api_key', str),
            'GMR_CHARACTER_MEMORY_MODEL': ('character_memory_model', str),
            'GMR_MAX_EVENT_HISTORY': ('max_event_history', int),
            'GMR_DIALOGUE_HISTORY_SIZE': ('dialogue_history_size', int),
            'GMR_DYNAMIC_FILTER_ENABLED': ('dynamic_filter_enabled', lambda v: str(v).lower() in {'1','true','yes','on'}),
            'GMR_FUZZY_MATCH_THRESHOLD': ('fuzzy_match_threshold', float),
            'GMR_MAX_PEOPLE_PRESENT': ('max_people_present', int),
            'GMR_MAX_NEARBY_SETTLEMENTS': ('max_nearby_settlements', int),
            'GMR_MAX_NEARBY_PARTIES': ('max_nearby_parties', int),
            'GMR_MAX_INVENTORY_LINES': ('max_inventory_lines', int),
            'GMR_MAX_EVENT_DIALOGUE_MESSAGES': ('max_event_dialogue_messages', int),
            'GMR_MAX_EVENT_DIALOGUE_SETTLEMENTS': ('max_event_dialogue_settlements', int),
            'GMR_LLM_LOG_ENABLED': ('llm_log_enabled', lambda v: str(v).lower() in {'1','true','yes','on'}),
            'GMR_LLM_LOG_PATH': ('llm_log_path', str),
            'GMR_LLM_LOG_PRETTY_JSON': ('llm_log_pretty_json', lambda v: str(v).lower() in {'1','true','yes','on'}),
            'GMR_SELECTOR_ENABLED': ('selector_enabled', lambda v: str(v).lower() in {'1','true','yes','on'}),
            'GMR_SELECTOR_API_URL': ('selector_api_url', str),
            'GMR_SELECTOR_API_KEY': ('selector_api_key', str),
            'GMR_SELECTOR_MODEL': ('selector_model', str),
            'GMR_SELECTOR_TEMPERATURE': ('selector_temperature', float),
            'GMR_SELECTOR_MAX_TOKENS': ('selector_max_tokens', int),
            'GMR_SELECTOR_TIMEOUT_SECONDS': ('selector_timeout_seconds', float),
            'GMR_SELECTOR_LOG_ENABLED': ('selector_log_enabled', lambda v: str(v).lower() in {'1','true','yes','on'}),
            'GMR_SELECTOR_LOG_PATH': ('selector_log_path', str),
            'GMR_SELECTOR_LOG_PRETTY_JSON': ('selector_log_pretty_json', lambda v: str(v).lower() in {'1','true','yes','on'}),
            'GMR_FILTERING_MODE': ('filtering_mode', str),
            'GMR_STATIC_GM_INDEX_SUMMARY_ENABLED': ('static_gm_index_summary_enabled', lambda v: str(v).lower() in {'1','true','yes','on'}),
            'GMR_STATIC_GM_INDEX_SUMMARY_MODEL': ('static_gm_index_summary_model', str),
            'GMR_STATIC_GM_INDEX_SUMMARY_API_URL': ('static_gm_index_summary_api_url', str),
            'GMR_STATIC_GM_INDEX_SUMMARY_API_KEY': ('static_gm_index_summary_api_key', str),
            'GMR_STATIC_GM_INDEX_SUMMARY_INSTRUCTION': ('static_gm_index_summary_instruction', str),
        }

        for env_key, (attr_name, converter) in env_mapping.items():
            if env_key in os.environ:
                setattr(self, attr_name, converter(os.environ[env_key]))

        if 'GMR_MODEL' in os.environ:
            model = os.environ['GMR_MODEL']
            for key in list(self.models.keys()):
                self.models[key] = model

        for key in list(self.models.keys()):
            env_key = f"GMR_MODEL_{key.upper()}"
            if env_key in os.environ:
                self.models[key] = os.environ[env_key]

    def save(self, path: Optional[str] = None):
        """Save current settings to JSON file."""
        if path is None:
            path = self.config_path or "config/settings.json"

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            'server': {
                'host': self.host,
                'port': self.port
            },
            'llm': {
                'api_url': self.api_url,
                'api_key': self.api_key,
                'models': self.models,
                'site_url': self.site_url,
                'app_title': self.app_title,
            },
            'request_type_detection': {
                'dialogue': self.request_type_signatures.get('dialogue', []),
                'events': self.request_type_signatures.get('events', []),
                'diplomacy': self.request_type_signatures.get('diplomacy', []),
            },
            'filtering': {
                'mode': self.get_filtering_mode(),
            },
            'gm': {
                'max_event_history': self.max_event_history,
                'dialogue_history_size': self.dialogue_history_size,
                'dynamic_filter_enabled': self.dynamic_filter_enabled,
                'fuzzy_match_threshold': self.fuzzy_match_threshold,
                'max_people_present': self.max_people_present,
                'max_nearby_settlements': self.max_nearby_settlements,
                'max_nearby_parties': self.max_nearby_parties,
                'max_inventory_lines': self.max_inventory_lines,
                'max_event_dialogue_messages': self.max_event_dialogue_messages,
                'max_event_dialogue_settlements': self.max_event_dialogue_settlements,
            },
            'prompt_drop_rules': self.prompt_drop_rules,
            'prompt_replace_rules': self.prompt_replace_rules,
            'request_parameters': self.request_parameters,
            'system_prompts': self.system_prompts,
            'llm_logging': {
                'enabled': self.llm_log_enabled,
                'path': self.llm_log_path,
                'pretty_json': self.llm_log_pretty_json,
            },
            'static_gm_index': {
                'enabled': self.static_gm_index_enabled,
                'ai_influence_folder': self.static_gm_index_ai_influence_folder,
                'files': self.static_gm_index_files,
                'db_path': self.static_gm_index_db_path,
                'selector_payload': self.static_gm_index_selector_payload,
                'summary_enabled': self.static_gm_index_summary_enabled,
                'summary_api_url': self.static_gm_index_summary_api_url,
                'summary_api_key': self.static_gm_index_summary_api_key,
                'summary_model': self.static_gm_index_summary_model,
                'summary_temperature': self.static_gm_index_summary_temperature,
                'summary_max_tokens': self.static_gm_index_summary_max_tokens,
                'summary_timeout_seconds': self.static_gm_index_summary_timeout_seconds,
                'summary_max_chars': self.static_gm_index_summary_max_chars,
                'summary_instruction': self.static_gm_index_summary_instruction,
            },
            'character_memory': {
                'enabled': self.character_memory_enabled,
                'campaign_dir': self.character_memory_campaign_dir,
                'api_url': self.character_memory_api_url,
                'api_key': self.character_memory_api_key,
                'model': self.character_memory_model,
                'temperature': self.character_memory_temperature,
                'max_tokens': self.character_memory_max_tokens,
                'timeout_seconds': self.character_memory_timeout_seconds,
                'preserve_last_lines': self.character_memory_preserve_last_lines,
                'auto_enabled': self.character_memory_auto_enabled,
                'auto_trigger_raw_lines': self.character_memory_auto_trigger_raw_lines,
                'auto_scan_interval_seconds': self.character_memory_auto_scan_interval_seconds,
                'auto_debounce_seconds': self.character_memory_auto_debounce_seconds,
                'max_memory_entries': self.character_memory_max_memory_entries,
                'summary_prompt': self.character_memory_summary_prompt,
                'profile_update_prompt': self.character_memory_profile_update_prompt,
            },
            'selector': {
                'enabled': self.selector_enabled,
                'api_url': self.selector_api_url,
                'api_key': self.selector_api_key,
                'model': self.selector_model,
                'temperature': self.selector_temperature,
                'max_tokens': self.selector_max_tokens,
                'timeout_seconds': self.selector_timeout_seconds,
                'instruction': self.selector_instruction,
                'context_rules': self.selector_context_rules,
                'log_enabled': self.selector_log_enabled,
                'log_path': self.selector_log_path,
                'pretty_json': self.selector_log_pretty_json,
            },
            'gui': {
                'log_viewer_enabled': self.gui_log_viewer_enabled,
            },
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def get_model(self, request_type: str = "events") -> str:
        """Get the model for a specific request type."""
        return self.models.get(request_type, self.models.get("events", next(iter(self.models.values()))))

    def get_enabled_request_parameters(self, request_type: str) -> Dict[str, Any]:
        """Return only enabled sampling parameters for the resolved request type."""
        request_type = request_type if request_type in REQUEST_TYPES else "events"
        params = self.request_parameters.get(request_type, {})
        outbound: Dict[str, Any] = {}
        for name in REQUEST_PARAMETER_NAMES:
            spec = params.get(name, {}) if isinstance(params, dict) else {}
            if isinstance(spec, dict) and _bool_from_any(spec.get("enabled", False)):
                value = _coerce_number(name, spec.get("value"))
                if value is not None:
                    outbound[name] = value
        return outbound

    def get_system_prompt_pair(self, request_type: str) -> tuple[str, str]:
        """Return optional pre/post system prompts for the resolved request type."""
        request_type = request_type if request_type in REQUEST_TYPES else "events"
        prompts = self.system_prompts.get(request_type, {})
        if not isinstance(prompts, dict):
            return "", ""
        return str(prompts.get("pre_history", "") or ""), str(prompts.get("post_history", "") or "")

    def get_chat_completions_url(self) -> str:
        """
        Return the final chat-completions endpoint.

        This accepts both styles in settings.json:
        - https://openrouter.ai/api/v1
        - https://openrouter.ai/api/v1/chat/completions
        """
        url = self.api_url.rstrip('/')
        if url.endswith('/chat/completions'):
            return url
        return f"{url}/chat/completions"

    def get_headers(self) -> Dict[str, str]:
        """Headers for the upstream LLM provider."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_title:
            headers["X-OpenRouter-Title"] = self.app_title
        return headers

    def get_filtering_mode(self) -> str:
        """Return the only supported policy-header filtering backend."""
        return DEFAULT_FILTERING_MODE

    def uses_selector_filtering(self) -> bool:
        return True

    def uses_static_gm_index(self) -> bool:
        return bool(self.static_gm_index_enabled)

    def to_dict(self) -> dict:
        """Export settings as dictionary."""
        return asdict(self)
