#!/usr/bin/env python3

# Copyright (C) 2026 SEGAREGA
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Modern GameMaster GUI.

This is a PyQt6 settings editor/server manager that preserves the full app
surface: connection settings, GM filtering, request parameters, system prompts,
selector configuration, detection rules, drop/replace rules, selector context
rules, and live log viewing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib import error, request


def ensure_console_streams() -> None:
    """Make stdout/stderr safe when running as a PyInstaller windowed EXE."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8", buffering=1)
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8", buffering=1)

try:
    from PyQt6.QtCore import QObject, QSize, QTimer, Qt, pyqtSignal
    from PyQt6.QtGui import QFont, QTextCursor
    from PyQt6.QtWidgets import (
        QAbstractSpinBox,
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QStyle,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover - runtime dependency message
    raise RuntimeError("PyQt6 is required for the modern GUI. Install it with: python -m pip install PyQt6") from exc


REQUEST_TYPES: Tuple[str, ...] = ("dialogue", "diplomacy", "events")
REQUEST_PARAMETERS: Tuple[str, ...] = ("temperature", "top_p", "top_k")
CONTEXT_LIMIT_POSITIONS: Tuple[Tuple[str, str], ...] = (
    ("Beginning", "beginning"),
    ("End", "end"),
    ("Beginning and end", "beginning_and_end"),
)

DEFAULT_SELECTOR_INSTRUCTION = (
    "You are an AI memory-filtering module for a roleplaying game. Your sole purpose is to select relevant lore and knowledge entries so an NPC can accurately respond to a Player.\n\n"
    "You will receive an input containing:\n"
    "1. CONTEXT EXTRACTS: The NPC's identity and background.\n"
    "2. Conversation History: The recent dialogue between the Player and the NPC.\n"
    "3. CANDIDATE INDEXED ENTRIES: A list of knowledge sections, each with an 'id', 'title', and 'Summary'.\n\n"
    "Instructions:\n"
    "1. Analyze the Player's most recent message(s) in the Conversation History.\n"
    "2. Read the Summary of each candidate entry.\n"
    "3. Determine if the Summary contains information the NPC realistically needs to know to formulate a coherent, contextually accurate reply to the Player.\n"
    "4. Extract the exact string from the `id` attribute (e.g., \"s001\", \"block_1\") of any relevant <SECTION> or <BLOCK>.\n\n"
    "Output Formatting Rules (CRITICAL):\n"
    "You must output strictly valid JSON and nothing else.\n"
    "- DO NOT wrap the output in markdown code blocks (e.g., no ```json or ```).\n"
    "- DO NOT include any conversational text, reasoning, greetings, or explanations.\n"
    "- DO NOT output anything outside of the JSON object.\n\n"
    "Your output must exactly match this schema:\n"
    "{\"blocks\":[{\"block_id\":\"block_1\",\"keep_ids\":[\"id1\",\"id2\"]}]}"
)
DEFAULT_FILTERING_MODE = "selector"
DEFAULT_SUMMARY_INSTRUCTION = (
    "You are an expert data compressor.\n\n"
    "Your task is to condense detailed information into ultra-concise, telegraphic summaries with an intent to reduce input-token usage for keyword database retrieval.\n\n"
    "Rules for Summarization:\n"
    "- No filler: Remove articles (a, an, the), flowery language, and full sentences.\n"
    "- Telegraphic style: Use slashes (/), semicolons (;), and commas to string together concepts tightly.\n"
    "- Prioritize the unique: Keep highly specific details and drop generic lore.\n"
    "- Use keywords, not sentences.\n"
    "- Format: [Name]: [Core Identity]. [Key details/customs/taboos]."
)

DEFAULT_CHARACTER_MEMORY_SUMMARY_PROMPT = (
    "You are compressing a character's conversation history for long-term game memory.\n\n"
    "Summarize only the provided conversation lines.\n\n"
    "Preserve:\n"
    "- Concrete facts, promises, threats, favors, secrets, relationships, conflicts, names, places, titles, and allegiances.\n"
    "- Emotional shifts, debts, bargains, and information the character learned from or about the player.\n"
    "- The character's attitude toward the player and any changes in trust, suspicion, respect, fear, anger, loyalty, or obligation.\n\n"
    "Do not invent events, motives, names, titles, or relationships.\n"
    "Do not contradict existing memory.\n"
    "Do not include trivial greetings, repeated phrasing, or generic banter unless it changed the relationship or revealed useful information.\n\n"
    "Write 1-2 short paragraphs in past tense. The paragraphs must be usable as a MEMORY entry inside ConversationHistory. Do not use markdown, bullets, headings, or JSON."
)

DEFAULT_CHARACTER_MEMORY_MERGE_PROMPT = (
    "Consolidate the provided MEMORY entries for one game character into one concise long-term memory paragraph suitable for a MEMORY entry in ConversationHistory.\n\n"
    "Preserve durable facts, names, places, titles, factions, relationships, promises, threats, secrets, debts, conflicts, favors, and changed attitudes toward the player.\n"
    "Preserve the character's latest known attitude and relationship state.\n"
    "When older and newer entries conflict, prefer the newest or most specific information.\n\n"
    "Remove repetition and trivial details.\n"
    "Do not invent anything.\n"
    "Do not use markdown, bullets, headings, labels, or JSON."
)

DEFAULT_CHARACTER_MEMORY_PROFILE_PROMPT = (
    "You are conservatively updating a game character profile using conversation history.\n\n"
    "You may update the character's personality or backstory only when the conversation reveals durable, meaningful information that should affect future roleplay.\n\n"
    "Examples include new relationships, loyalties, grudges, debts, promises, losses, family news, imprisonment, release, betrayal, alliance, fear, respect, or changed opinion of the player.\n\n"
    "Do not rewrite the character into a different person.\n"
    "Preserve their established temperament, social status, history, culture, speech style, values, and contradictions unless the conversation gives strong evidence of gradual change.\n"
    "Do not turn a cruel character kind, a cynical character trusting, a noble-born character lowborn, or a lifelong enemy into a friend without strong evidence.\n"
    "Prefer small additive edits over broad rewrites.\n"
    "Keep the existing structure and style where possible.\n\n"
    "If no meaningful durable change is needed, return changed=false.\n\n"
    "Return strict JSON only with this shape:\n"
    '{"changed": true or false, "new_personality": string or null, "new_backstory": string or null, "reason": string, "confidence": "low" | "medium" | "high"}.'
)
HIDDEN_GM_FIELDS = {
    "max" + "_sections",
    "min" + "_similarity",
    "use" + "_rag",
    "rag" + "_enabled",
    "embedding" + "_model",
    "dialogue" + "_history" + "_size",
}

DEFAULT_STATIC_GM_INDEX_FILES = [
    "world.txt",
    "actionrules.txt",
]

KNOWN_GM_FIELDS: Dict[str, Tuple[str, str, Any, Any]] = {
    "dynamic_filter_enabled": ("Enable dynamic GM filtering", "bool", None, True),
    "disable_user_last_message_during_npc_npc_conversation": ("Disable User's last message during NPC-NPC conversation", "bool", None, False),
    "disable_user_last_message_during_group_chat": ("Disable User's last message during Group Chat", "bool", None, False),
    "max_event_history": ("Max event history", "int", (0, 1000), 200),
    "fuzzy_match_threshold": ("Dynamic data fuzzy match threshold", "float", (0.0, 1.0), 0.88),
    "max_people_present": ("Max people present", "int", (0, 500), 10),
    "max_nearby_settlements": ("Max nearby settlements", "int", (0, 500), 7),
    "max_nearby_parties": ("Max nearby parties", "int", (0, 500), 5),
    "max_inventory_lines": ("Max inventory lines", "int", (0, 500), 5),
    "max_event_dialogue_messages": ("Events - max dialogue lines", "int", (0, 500), 20),
    "max_event_dialogue_settlements": ("Events - max settlements mentioned", "int", (0, 500), 10),
}

DYNAMIC_HIDE_UNTIL_RELEVANT_DEFAULTS: Dict[str, bool] = {
    "character_briefing": False,
    "player_current_data": False,
    "people_present": False,
    "nearby_settlements": False,
    "nearby_parties": False,
    "mentioned_settlements": False,
    "mentioned_characters": False,
    "mentioned_parties": False,
    "appearance_equipment": False,
    "wealth_money": False,
    "inventory_items": False,
    "clan": False,
    "family_relatives": False,
    "relations": False,
    "forces": False,
    "captives": False,
    "workshops": False,
}

DYNAMIC_HIDE_UNTIL_RELEVANT_LABELS: Tuple[Tuple[str, str], ...] = (
    ("character_briefing", "Character Briefing (CURRENT DATA)"),
    ("player_current_data", "The Player Current Data"),
    ("people_present", "People physically present"),
    ("nearby_settlements", "Nearby settlements"),
    ("nearby_parties", "Nearby parties"),
    ("mentioned_settlements", "Mentioned settlements"),
    ("mentioned_characters", "Mentioned characters"),
    ("mentioned_parties", "Mentioned parties"),
    ("appearance_equipment", "Appearance/equipment lines"),
    ("wealth_money", "Wealth/money lines"),
    ("inventory_items", "Inventory/item lines"),
    ("clan", "Clan line"),
    ("family_relatives", "Family/relatives lines"),
    ("relations", "Relations/friends/enemies lines"),
    ("forces", "Forces/troops lines"),
    ("captives", "Captives/prisoners lines"),
    ("workshops", "Workshops/business lines"),
)


# ---------------------------------------------------------------------------
# Path and config helpers
# ---------------------------------------------------------------------------

def app_root() -> Path:
    """Return the writable app folder.

    In PyInstaller --onefile mode, bundled resources live in a temporary
    extraction folder, but user-editable files must live next to the EXE.
    So frozen builds always use the directory containing sys.executable.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_ROOT = app_root()
SETTINGS_PATH = APP_ROOT / "config" / "settings.json"


def default_request_parameters() -> Dict[str, Dict[str, Dict[str, Any]]]:
    return {
        request_type: {
            "temperature": {"enabled": False, "value": 1.0},
            "top_p": {"enabled": False, "value": 1.0},
            "top_k": {"enabled": False, "value": 40},
        }
        for request_type in REQUEST_TYPES
    }


def default_request_type_detection() -> Dict[str, List[str]]:
    return {
        "dialogue": [
            "### Mission ###\nRole-play as a character in Mount & Blade II: Bannerlord. Use your personality, history, and context to inform responses. Output ONLY a valid JSON object with no extra text or markdown.",
            "===== GROUP CONVERSATION MODE =====",
            "===== NPC-TO-NPC CONVERSATION MODE =====",
        ],
        "events": [
            "## EVENT STRUCTURE:\nMUST include: 1) CAUSE (from data) 2) ACTION (decision taken) 3) CONSEQUENCE (future impact)\nPrefer DEVELOPING existing conflicts over new minor incidents. Return [] if insufficient data.",
        ],
        "diplomacy": [
            "### CRITICAL REMINDER: You Are a Living Ruler ###",
        ],
    }


def default_selector_context_rules() -> List[Dict[str, Any]]:
    return [
        {
            "name": "Character Briefing (CURRENT DATA)",
            "request_types": ["dialogue"],
            "beginning": "### Character Briefing (CURRENT DATA) ###",
            "end": "**Description:**",
            "include_beginning_marker": False,
            "include_end_marker": False,
            "limit_enabled": False,
            "limit_chars": 5000,
            "limit_position": "end",
        },
        {
            "name": "### Conversation History ###",
            "request_types": ["dialogue"],
            "beginning": "### Conversation History ###",
            "end": "Last Interaction:",
            "include_beginning_marker": False,
            "include_end_marker": False,
            "limit_enabled": False,
            "limit_chars": 5000,
            "limit_position": "end",
        },
    ]


def default_system_prompts() -> Dict[str, Dict[str, str]]:
    return {request_type: {"pre_history": "", "post_history": ""} for request_type in REQUEST_TYPES}


def default_config() -> Dict[str, Any]:
    return {
        "server": {"host": "localhost", "port": 5100},
        "llm": {
            "api_url": "https://api.openai.com/v1",
            "api_key": "",
            "models": {"events": "gpt-4-turbo", "diplomacy": "gpt-4-turbo", "dialogue": "gpt-4-turbo"},
            "site_url": "",
            "app_title": "GameMaster",
        },
        "request_type_detection": default_request_type_detection(),
        "filtering": {"mode": DEFAULT_FILTERING_MODE},
        "gm": {key: info[3] for key, info in KNOWN_GM_FIELDS.items()},
        "static_gm_index": {
            "enabled": True,
            "ai_influence_folder": "",
            "files": list(DEFAULT_STATIC_GM_INDEX_FILES),
            "db_path": "cache/static_gm_index.sqlite3",
            # Reindexing is manual-only; do not start summary/index calls on reload/startup.
            "selector_payload": "summary",
            "summary_enabled": True,
            "summary_api_url": "",
            "summary_api_key": "",
            "summary_model": "",
            "summary_temperature": 0.1,
            "summary_max_tokens": 220,
            "summary_timeout_seconds": 120.0,
            "summary_max_chars": 6000,
            "summary_instruction": DEFAULT_SUMMARY_INSTRUCTION,
        },
        "character_memory": {
            "enabled": True,
            "campaign_dir": "",
            "api_url": "",
            "api_key": "",
            "model": "",
            "temperature": 0.1,
            "max_tokens": 32000,
            "timeout_seconds": 180.0,
            "preserve_last_lines": 20,
            "auto_enabled": False,
            "auto_trigger_raw_lines": 30,
            "auto_scan_interval_seconds": 30.0,
            "auto_debounce_seconds": 8.0,
            "max_memory_entries": 5,
            "summary_prompt": DEFAULT_CHARACTER_MEMORY_SUMMARY_PROMPT,
            "merge_prompt": DEFAULT_CHARACTER_MEMORY_MERGE_PROMPT,
            "profile_update_prompt": DEFAULT_CHARACTER_MEMORY_PROFILE_PROMPT,
        },
        "prompt_drop_rules": [],
        "prompt_replace_rules": [],
        "llm_logging": {"enabled": False, "path": "logs/llm-log.txt", "pretty_json": True},
        "selector": {
            "enabled": True,
            "api_url": "",
            "api_key": "",
            "model": "",
            "temperature": 0.0,
            "max_tokens": 32000,
            "timeout_seconds": 120.0,
            "instruction": DEFAULT_SELECTOR_INSTRUCTION,
            "log_enabled": False,
            "log_path": "logs/selector-log.txt",
            "pretty_json": True,
            "context_rules": default_selector_context_rules(),
        },
        "request_parameters": default_request_parameters(),
        "system_prompts": default_system_prompts(),
        "gui": {"log_viewer_enabled": False},
    }


def merge_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(config) if isinstance(config, dict) else {}
    defaults = default_config()

    def merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
        for key, val in src.items():
            if key not in dst:
                dst[key] = deepcopy(val)
            elif isinstance(dst[key], dict) and isinstance(val, dict):
                merge_dict(dst[key], val)
        return dst

    merge_dict(cfg, defaults)

    if "gm" not in cfg or not isinstance(cfg.get("gm"), dict):
        cfg["gm"] = {}
    merge_dict(cfg["gm"], defaults["gm"])
    cfg.pop("retr" + "ieval", None)

    static_index = cfg.setdefault("static_gm_index", {})
    if isinstance(static_index, dict) and not str(static_index.get("summary_instruction", "") or "").strip():
        static_index["summary_instruction"] = DEFAULT_SUMMARY_INSTRUCTION

    if "filtering" not in cfg or not isinstance(cfg.get("filtering"), dict):
        cfg["filtering"] = {"mode": DEFAULT_FILTERING_MODE}
    cfg["filtering"]["mode"] = DEFAULT_FILTERING_MODE

    # Prompt drop/replace rules live only at the top level.  Older nested gm copies
    # are removed so settings.json does not contain duplicate rule lists.
    gm = cfg.get("gm", {}) if isinstance(cfg.get("gm"), dict) else {}
    if isinstance(gm, dict):
        if not cfg.get("prompt_drop_rules") and isinstance(gm.get("prompt_drop_rules"), list):
            cfg["prompt_drop_rules"] = deepcopy(gm.get("prompt_drop_rules"))
        if not cfg.get("prompt_replace_rules") and isinstance(gm.get("prompt_replace_rules"), list):
            cfg["prompt_replace_rules"] = deepcopy(gm.get("prompt_replace_rules"))
        gm.pop("prompt_drop_rules", None)
        gm.pop("prompt_replace_rules", None)

    # Normalize request parameter and prompt blocks.
    params = cfg.setdefault("request_parameters", {})
    for request_type in REQUEST_TYPES:
        params.setdefault(request_type, {})
        for name, default in default_request_parameters()[request_type].items():
            raw = params[request_type].get(name, deepcopy(default))
            if isinstance(raw, dict):
                params[request_type][name] = {
                    "enabled": bool(raw.get("enabled", False)),
                    "value": raw.get("value", default["value"]),
                }
            else:
                params[request_type][name] = {"enabled": raw is not None, "value": raw if raw is not None else default["value"]}

    prompts = cfg.setdefault("system_prompts", {})
    for request_type in REQUEST_TYPES:
        prompts.setdefault(request_type, {})
        prompts[request_type].setdefault("pre_history", "")
        prompts[request_type].setdefault("post_history", "")

    return cfg


def write_default_settings_if_missing(path: Path) -> Dict[str, Any]:
    """Create a real settings file next to the EXE/source tree on first launch."""
    cfg = merge_defaults({})
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"Could not create default settings at {path}: {exc}")
    return cfg


def load_json_settings(path: Path) -> Dict[str, Any]:
    candidates = [path, APP_ROOT / "settings.example.json", APP_ROOT / "config" / "settings - default.json"]
    for candidate in candidates:
        if candidate.exists():
            try:
                return merge_defaults(json.loads(candidate.read_text(encoding="utf-8")))
            except Exception as exc:
                print(f"Could not load {candidate}: {exc}")
    return write_default_settings_if_missing(path)


def text_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def path_from_text(raw_path: str) -> Path:
    path = Path(raw_path.strip())
    return path if path.is_absolute() else APP_ROOT / path


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

class Card(QFrame):
    def __init__(self, title: str = "", subtitle: str = "") -> None:
        super().__init__()
        self.setObjectName("Card")
        self.vbox = QVBoxLayout(self)
        self.vbox.setContentsMargins(12, 12, 12, 12)
        self.vbox.setSpacing(8)
        if title:
            label = QLabel(title)
            label.setObjectName("CardTitle")
            self.vbox.addWidget(label)
        if subtitle:
            label = QLabel(subtitle)
            label.setObjectName("Hint")
            label.setWordWrap(True)
            self.vbox.addWidget(label)


def make_scroll_page() -> Tuple[QScrollArea, QWidget, QVBoxLayout]:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(6, 6, 6, 16)
    layout.setSpacing(10)
    scroll.setWidget(page)
    return scroll, page, layout


def polish_line_edit(widget: QLineEdit) -> QLineEdit:
    widget.setMinimumHeight(30)
    widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return widget


def polish_text_edit(widget: QPlainTextEdit, height: int = 90) -> QPlainTextEdit:
    widget.setMinimumHeight(height)
    widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
    return widget


def make_spin(kind: str, value: Any, bounds: Optional[Tuple[Any, Any]] = None) -> QSpinBox | QDoubleSpinBox:
    if kind == "float":
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setSingleStep(0.01)
        low, high = bounds or (-999999.0, 999999.0)
        spin.setRange(float(low), float(high))
        try:
            spin.setValue(float(value))
        except Exception:
            spin.setValue(0.0)
    else:
        spin = QSpinBox()
        low, high = bounds or (-999999, 999999)
        spin.setRange(int(low), int(high))
        try:
            spin.setValue(int(value))
        except Exception:
            spin.setValue(0)
    spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    spin.setMinimumHeight(30)
    spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return spin


def format_request_parameter_value(name: str, value: Any) -> str:
    default = 40 if name == "top_k" else 1.0
    if value is None or value == "":
        value = default
    if name == "top_k":
        try:
            return str(int(value))
        except Exception:
            return str(default)
    try:
        return (f"{float(value):.6g}")
    except Exception:
        return f"{default:.6g}"


def parse_request_parameter_value(name: str, text: str) -> int | float:
    cleaned = str(text).strip()
    if cleaned == "":
        return 40 if name == "top_k" else 1.0
    if name == "top_k":
        return int(float(cleaned))
    return float(cleaned)


def add_grid_row(grid: QGridLayout, row: int, label_text: str, widget: QWidget, hint: str = "") -> None:
    label = QLabel(label_text)
    label.setObjectName("FieldLabel")
    label.setMinimumWidth(160)
    label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
    grid.addWidget(label, row, 0)
    grid.addWidget(widget, row, 1)
    if hint:
        hint_label = QLabel(hint)
        hint_label.setObjectName("TinyHint")
        hint_label.setWordWrap(True)
        grid.addWidget(hint_label, row + 1, 1)


def plain_text(text_edit: QPlainTextEdit) -> str:
    return text_edit.toPlainText()


PROMPT_RULE_TEXT_KEYS = {
    "drop_beginning",
    "drop_end",
    "replace_beginning",
    "replace_end",
    "replacement_text",
    "scope_beginning",
    "scope_end",
    "beginning",
    "end",
}


def normalize_prompt_rule_text(value: Any) -> str:
    """Normalize GUI-entered prompt-rule text to the real marker text.

    Users often copy marker strings from settings.json, where JSON requires quotes
    to appear as \". The GUI fields should store the real prompt text, not those
    JSON escape characters; otherwise saving escapes the backslashes again and the
    config becomes {\\\"...}. Only decode when the value clearly contains escaped
    JSON quotes, so ordinary text and literal backslashes are preserved.
    """
    text = str(value or "")
    if '\"' not in text:
        return text
    try:
        decoded = json.loads('"' + text + '"')
    except Exception:
        return text
    return decoded if isinstance(decoded, str) else text


# ---------------------------------------------------------------------------
# Model fetch helpers
# ---------------------------------------------------------------------------

class SignalBus(QObject):
    models_fetched = pyqtSignal(str, list, str)
    status = pyqtSignal(str, str)
    reindex_button_enabled = pyqtSignal(bool)
    character_memory_buttons_enabled = pyqtSignal(bool)
    character_memory_result = pyqtSignal(object)
    start_server_requested = pyqtSignal()


def make_model_combo() -> QComboBox:
    combo = QComboBox()
    combo.setEditable(True)
    combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    combo.setMinimumHeight(30)
    combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return combo


def combo_text(combo: QComboBox) -> str:
    return combo.currentText().strip()


def set_combo_text(combo: QComboBox, text: str) -> None:
    text = str(text or "")
    if text and combo.findText(text) < 0:
        combo.addItem(text)
    combo.setCurrentText(text)


def normalize_models_url(api_url: str) -> str:
    base = (api_url or "").strip()
    if not base:
        raise ValueError("API URL is empty.")
    base = base.rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/responses"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base.lower().endswith("/models"):
        return base
    return f"{base}/models"


def parse_model_ids(payload: Any) -> List[str]:
    if isinstance(payload, dict):
        data = payload.get("data", [])
        if isinstance(data, dict) and isinstance(data.get("models"), list):
            data = data.get("models", [])
    elif isinstance(payload, list):
        data = payload
    else:
        data = []
    models: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                model_id = item.strip()
            elif isinstance(item, dict):
                model_id = str(item.get("id") or item.get("name") or item.get("model") or "").strip()
            else:
                model_id = ""
            if model_id and model_id not in models:
                models.append(model_id)
    return sorted(models, key=lambda x: x.lower())

# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class GameMasterGUI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings_path = SETTINGS_PATH
        self.settings: Dict[str, Any] = load_json_settings(self.settings_path)
        self.server_process: Optional[subprocess.Popen[Any]] = None
        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self.refresh_logs)
        self.signals = SignalBus()
        self.signals.models_fetched.connect(self.apply_fetched_models)
        self.signals.status.connect(self.set_status)
        self.signals.start_server_requested.connect(self.start_server)
        self.signals.character_memory_buttons_enabled.connect(self.set_character_memory_buttons_enabled)
        self.signals.character_memory_result.connect(self.write_character_memory_result)

        self.request_parameter_widgets: Dict[str, Dict[str, Tuple[QCheckBox, QLineEdit]]] = {}
        self.system_prompt_widgets: Dict[str, Dict[str, QPlainTextEdit]] = {}
        self.gm_widgets: Dict[str, QWidget] = {}
        self.gm_dynamic_hide_widgets: Dict[str, QCheckBox] = {}
        self.gm_extra_widgets: Dict[str, QWidget] = {}
        self.model_fields: Dict[str, QComboBox] = {}
        self.detection_rules: List[Dict[str, Any]] = []
        self.detection_widgets: List[Dict[str, QWidget]] = []
        self.drop_rules: List[Dict[str, Any]] = []
        self.replace_rules: List[Dict[str, Any]] = []
        self.context_rules: List[Dict[str, Any]] = []
        self.rule_widgets: Dict[str, List[Dict[str, QWidget]]] = {}

        self.setWindowTitle("GameMaster Control Center")
        self.resize(1160, 760)
        self.setMinimumSize(860, 560)
        self.build_ui()
        self.apply_config_to_ui()
        self.log_timer.setInterval(1000)
        # Preserve original app behavior: try to start after UI is visible.
        QTimer.singleShot(250, self.start_server)

    # ------------------------------------------------------------------
    # Build shell
    # ------------------------------------------------------------------
    def build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        header = QFrame()
        header.setObjectName("HeaderBar")
        h = QHBoxLayout(header)
        h.setContentsMargins(12, 10, 12, 10)
        h.setSpacing(8)
        title_box = QVBoxLayout()
        title = QLabel("GameMaster")
        title.setObjectName("Title")
        subtitle = QLabel("Settings editor, prompt rules, request controls, model picker, and logs.")
        subtitle.setObjectName("Subtitle")
        subtitle.setWordWrap(True)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        h.addLayout(title_box, 1)

        self.start_btn = QPushButton("Start")
        self.start_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.start_btn.clicked.connect(self.start_server)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.clicked.connect(self.stop_server)
        self.save_btn = QPushButton("Save & Reload")
        self.save_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.save_btn.setObjectName("PrimaryButton")
        self.save_btn.clicked.connect(self.save_settings)
        self.open_btn = QPushButton("Settings…")
        self.open_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.open_btn.clicked.connect(self.choose_settings_file)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusPill")
        for btn in (self.start_btn, self.stop_btn, self.save_btn, self.open_btn):
            btn.setIconSize(QSize(16, 16))
            btn.setMinimumHeight(32)
        h.addWidget(self.open_btn)
        h.addWidget(self.start_btn)
        h.addWidget(self.stop_btn)
        h.addWidget(self.save_btn)
        h.addWidget(self.status_label)
        outer.addWidget(header)

        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setElideMode(Qt.TextElideMode.ElideRight)
        self.tabs.addTab(self.build_connection_tab(), "General")
        self.tabs.addTab(self.build_gm_filtering_tab(), "GM Filtering")
        self.tabs.addTab(self.build_request_parameters_tab(), "Request Parameters")
        self.tabs.addTab(self.build_system_prompts_tab(), "System Prompts")
        self.tabs.addTab(self.build_selector_tab(), "Selector")
        self.tabs.addTab(self.build_detection_tab(), "Detection")
        self.tabs.addTab(self.build_rules_tab("drop", "Prompt Drop Rules", self.drop_rule_schema()), "Drop Rules")
        self.tabs.addTab(self.build_rules_tab("replace", "Prompt Replace Rules", self.replace_rule_schema()), "Replace Rules")
        self.tabs.addTab(self.build_rules_tab("context", "Selector Context Rules", self.context_rule_schema()), "Context Rules")
        self.tabs.addTab(self.build_character_memory_tab(), "Character Memory")
        self.tabs.addTab(self.build_log_viewer_tab(), "Log Viewer")
        outer.addWidget(self.tabs, 1)
        self.setCentralWidget(root)

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------
    def build_connection_tab(self) -> QWidget:
        scroll, page, layout = make_scroll_page()

        server_card = Card("Server", "Host and port used by the local proxy.")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.host_entry = polish_line_edit(QLineEdit())
        self.port_entry = make_spin("int", 5100, (1, 65535))
        add_grid_row(grid, 0, "Host", self.host_entry)
        add_grid_row(grid, 1, "Port", self.port_entry)
        server_card.vbox.addLayout(grid)
        layout.addWidget(server_card)

        llm_card = Card("LLM Backend", "Main upstream API configuration.")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.api_url_entry = polish_line_edit(QLineEdit())
        self.api_key_entry = polish_line_edit(QLineEdit())
        self.api_key_entry.setEchoMode(QLineEdit.EchoMode.Password)
        self.site_url_entry = polish_line_edit(QLineEdit())
        self.app_title_entry = polish_line_edit(QLineEdit())
        add_grid_row(grid, 0, "API URL", self.api_url_entry)
        add_grid_row(grid, 1, "API Key", self.api_key_entry)
        add_grid_row(grid, 2, "Site URL / Referer", self.site_url_entry)
        add_grid_row(grid, 3, "App Title", self.app_title_entry)
        llm_card.vbox.addLayout(grid)
        layout.addWidget(llm_card)

        models_card = Card("Models by Request Type", "Pick from fetched models or type a model ID manually.")
        fetch_row = QHBoxLayout()
        self.fetch_main_models_btn = QPushButton("Fetch Models")
        self.fetch_main_models_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.fetch_main_models_btn.setIconSize(QSize(16, 16))
        self.fetch_main_models_btn.clicked.connect(lambda: self.fetch_models("main"))
        fetch_row.addWidget(QLabel("Fetches from API URL + /models and fills all three editable model boxes."))
        fetch_row.addStretch(1)
        fetch_row.addWidget(self.fetch_main_models_btn)
        models_card.vbox.addLayout(fetch_row)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        for row, request_type in enumerate(REQUEST_TYPES):
            field = make_model_combo()
            self.model_fields[request_type] = field
            add_grid_row(grid, row, request_type.title(), field)
        models_card.vbox.addLayout(grid)
        layout.addWidget(models_card)

        logging_card = Card("Logging", "Live viewer and request/selector log files. The actual log windows are on the Log Viewer tab.")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.log_viewer_enabled_var = QCheckBox("Enable live log viewer")
        self.llm_log_enabled_var = QCheckBox("Enabled")
        self.llm_log_path_entry = polish_line_edit(QLineEdit())
        self.llm_log_pretty_var = QCheckBox("Pretty JSON")
        self.sel_log_enabled_var = QCheckBox("Enabled")
        self.sel_log_path_entry = polish_line_edit(QLineEdit())
        self.sel_log_pretty_var = QCheckBox("Pretty JSON")

        llm_path_row = QWidget()
        llm_path_layout = QHBoxLayout(llm_path_row)
        llm_path_layout.setContentsMargins(0, 0, 0, 0)
        llm_path_layout.setSpacing(8)
        llm_path_layout.addWidget(self.llm_log_path_entry, 1)
        browse_llm_log = QPushButton("Browse…")
        browse_llm_log.clicked.connect(lambda: self.choose_log_file(self.llm_log_path_entry))
        llm_path_layout.addWidget(browse_llm_log)

        selector_path_row = QWidget()
        selector_path_layout = QHBoxLayout(selector_path_row)
        selector_path_layout.setContentsMargins(0, 0, 0, 0)
        selector_path_layout.setSpacing(8)
        selector_path_layout.addWidget(self.sel_log_path_entry, 1)
        browse_selector_log = QPushButton("Browse…")
        browse_selector_log.clicked.connect(lambda: self.choose_log_file(self.sel_log_path_entry))
        selector_path_layout.addWidget(browse_selector_log)

        add_grid_row(grid, 0, "Live log viewer", self.log_viewer_enabled_var)
        add_grid_row(grid, 1, "LLM logging", self.llm_log_enabled_var)
        add_grid_row(grid, 2, "LLM log path", llm_path_row)
        add_grid_row(grid, 3, "LLM pretty JSON", self.llm_log_pretty_var)
        add_grid_row(grid, 4, "Selector logging", self.sel_log_enabled_var)
        add_grid_row(grid, 5, "Selector log path", selector_path_row)
        add_grid_row(grid, 6, "Selector pretty JSON", self.sel_log_pretty_var)
        logging_card.vbox.addLayout(grid)
        layout.addWidget(logging_card)

        self.log_viewer_enabled_var.toggled.connect(self.on_log_viewer_toggled)
        self.llm_log_path_entry.textChanged.connect(lambda *_: self.refresh_logs())
        self.sel_log_path_entry.textChanged.connect(lambda *_: self.refresh_logs())

        layout.addStretch(1)
        return scroll

    def build_gm_filtering_tab(self) -> QWidget:
        scroll, page, layout = make_scroll_page()
        note = QLabel("Use [GM] headers for selector-controlled content, [PINNED] for always-included content, and [IGNORE] for content that should be removed.")
        note.setObjectName("Hint")
        note.setWordWrap(True)
        layout.addWidget(note)

        core = Card("Core GM Filtering", "Filtering behavior and history thresholds from settings.json → gm.")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        row = 0
        for key in (
            "dynamic_filter_enabled",
            "disable_user_last_message_during_npc_npc_conversation",
            "disable_user_last_message_during_group_chat",
            "fuzzy_match_threshold",
            "max_event_history",
            "max_event_dialogue_messages",
            "max_event_dialogue_settlements",
        ):
            label, kind, bounds, default = KNOWN_GM_FIELDS[key]
            widget = self.make_setting_widget(kind, default, bounds)
            self.gm_widgets[key] = widget
            add_grid_row(grid, row, label, widget)
            row += 1
        core.vbox.addLayout(grid)
        layout.addWidget(core)

        limits = Card("Dynamic Context Limits", "Caps for dynamically selected people, settlements, parties, and inventory.")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        row = 0
        for key in (
            "max_people_present",
            "max_nearby_settlements",
            "max_nearby_parties",
            "max_inventory_lines",
        ):
            label, kind, bounds, default = KNOWN_GM_FIELDS[key]
            widget = self.make_setting_widget(kind, default, bounds)
            self.gm_widgets[key] = widget
            add_grid_row(grid, row, label, widget)
            row += 1
        limits.vbox.addLayout(grid)
        layout.addWidget(limits)

        dynamic_controls = Card(
            "Hide-Until-Relevant Controls",
            "Checked means the proxy may hide or summarize that live AIInfluence section until the current request makes it relevant. Unchecked means pass that section/line through unfiltered."
        )
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        for row, (key, label) in enumerate(DYNAMIC_HIDE_UNTIL_RELEVANT_LABELS):
            widget = QCheckBox("Hide until relevant")
            self.gm_dynamic_hide_widgets[key] = widget
            add_grid_row(grid, row, label, widget)
        dynamic_controls.vbox.addLayout(grid)
        layout.addWidget(dynamic_controls)

        self.gm_extras_card = Card("Additional GM Keys", "Any simple extra keys already present under settings.json → gm are editable here and preserved on save.")
        self.gm_extras_layout = QGridLayout()
        self.gm_extras_layout.setColumnStretch(1, 1)
        self.gm_extras_card.vbox.addLayout(self.gm_extras_layout)
        layout.addWidget(self.gm_extras_card)

        layout.addStretch(1)
        return scroll

    def build_request_parameters_tab(self) -> QWidget:
        scroll, page, layout = make_scroll_page()
        intro = QLabel("Unchecked parameters are omitted from outbound requests. Checked parameters are sent only for that request type.")
        intro.setObjectName("Hint")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        for request_type in REQUEST_TYPES:
            card = Card(request_type.title(), "Configure optional sampling parameters.")
            grid = QGridLayout()
            grid.setColumnStretch(2, 1)
            grid.addWidget(QLabel("Send"), 0, 0)
            grid.addWidget(QLabel("Parameter"), 0, 1)
            grid.addWidget(QLabel("Value"), 0, 2)
            self.request_parameter_widgets[request_type] = {}
            for row, name in enumerate(REQUEST_PARAMETERS, start=1):
                checkbox = QCheckBox()
                field = polish_line_edit(QLineEdit(format_request_parameter_value(name, 40 if name == "top_k" else 1.0)))
                if name == "temperature":
                    field.setPlaceholderText("e.g. 1 or 1.2")
                    hint = "OpenAI-compatible float; commonly 0-2."
                elif name == "top_p":
                    field.setPlaceholderText("e.g. 1 or 0.9")
                    hint = "OpenAI-compatible float; commonly 0-1."
                else:
                    field.setPlaceholderText("e.g. 40")
                    hint = "Provider-specific integer. Leave unchecked for strict OpenAI."
                field.setToolTip(hint)
                field.setEnabled(False)
                checkbox.toggled.connect(field.setEnabled)
                label = QLabel(name)
                label.setObjectName("FieldLabel")
                label.setToolTip(hint)
                grid.addWidget(checkbox, row, 0, Qt.AlignmentFlag.AlignCenter)
                grid.addWidget(label, row, 1)
                grid.addWidget(field, row, 2)
                self.request_parameter_widgets[request_type][name] = (checkbox, field)
            card.vbox.addLayout(grid)
            layout.addWidget(card)
        layout.addStretch(1)
        return scroll

    def build_system_prompts_tab(self) -> QWidget:
        scroll, page, layout = make_scroll_page()
        intro = QLabel("Pre-history is inserted as the first system prompt. Post-history is inserted as the last system prompt. Empty fields are ignored.")
        intro.setObjectName("Hint")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        nested = QTabWidget()
        nested.setObjectName("InnerTabs")
        for request_type in REQUEST_TYPES:
            tab = QWidget()
            v = QVBoxLayout(tab)
            v.setContentsMargins(4, 12, 4, 4)
            v.setSpacing(12)
            card = Card(request_type.title())
            pre = polish_text_edit(QPlainTextEdit(), 130)
            pre.setPlaceholderText(f"Optional FIRST system prompt for {request_type} requests")
            post = polish_text_edit(QPlainTextEdit(), 130)
            post.setPlaceholderText(f"Optional LAST system prompt for {request_type} requests")
            pre_label = QLabel("Pre-history system prompt")
            pre_label.setObjectName("FieldLabel")
            post_label = QLabel("Post-history system prompt")
            post_label.setObjectName("FieldLabel")
            card.vbox.addWidget(pre_label)
            card.vbox.addWidget(pre)
            card.vbox.addWidget(post_label)
            card.vbox.addWidget(post)
            self.system_prompt_widgets[request_type] = {"pre_history": pre, "post_history": post}
            v.addWidget(card)
            v.addStretch(1)
            nested.addTab(tab, request_type.title())
        layout.addWidget(nested, 1)
        return scroll

    def build_selector_tab(self) -> QWidget:
        scroll, page, layout = make_scroll_page()

        selector_note = Card("Selector", "Selector mode is the only filtering backend. Static GM Index sends all [GM] child summaries to the selector, then the selector returns the exact IDs to inject in full.")
        note = QLabel("Only selector/static-GM-index filtering is used. [PINNED] is deterministic, [IGNORE] is removed, and [GM] is summary-selected.")
        note.setObjectName("Hint")
        note.setWordWrap(True)
        selector_note.vbox.addWidget(note)
        layout.addWidget(selector_note)

        config_card = Card("Selector Model", "Used for [GM] section decisions and, by default, one-time LLM summaries during reindex.")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.sel_api_url_entry = polish_line_edit(QLineEdit())
        self.sel_api_key_entry = polish_line_edit(QLineEdit())
        self.sel_api_key_entry.setEchoMode(QLineEdit.EchoMode.Password)
        self.sel_model_entry = make_model_combo()
        self.sel_temp_entry = polish_line_edit(QLineEdit(format_request_parameter_value("temperature", 0.0)))
        self.sel_temp_entry.setPlaceholderText("e.g. 0, 0.7, 1, or 1.2")
        self.sel_temp_entry.setToolTip("Selector sampling temperature. Saved as a plain JSON number; no forced 0.000 display.")
        self.sel_max_tokens_entry = make_spin("int", 32000, (1, 1000000))
        self.sel_timeout_entry = make_spin("float", 120.0, (1.0, 3600.0))
        add_grid_row(grid, 0, "API URL", self.sel_api_url_entry)
        add_grid_row(grid, 1, "API Key", self.sel_api_key_entry)
        selector_model_row = QWidget()
        selector_model_layout = QHBoxLayout(selector_model_row)
        selector_model_layout.setContentsMargins(0, 0, 0, 0)
        selector_model_layout.setSpacing(8)
        selector_model_layout.addWidget(self.sel_model_entry, 1)
        self.fetch_selector_models_btn = QPushButton("Fetch Models")
        self.fetch_selector_models_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.fetch_selector_models_btn.setIconSize(QSize(16, 16))
        self.fetch_selector_models_btn.clicked.connect(lambda: self.fetch_models("selector"))
        selector_model_layout.addWidget(self.fetch_selector_models_btn)
        add_grid_row(grid, 2, "Model", selector_model_row)
        add_grid_row(grid, 3, "Temperature", self.sel_temp_entry)
        add_grid_row(grid, 4, "Max tokens", self.sel_max_tokens_entry)
        add_grid_row(grid, 5, "Timeout seconds", self.sel_timeout_entry)
        config_card.vbox.addLayout(grid)
        layout.addWidget(config_card)

        instruction_card = Card("Selector Instruction")
        self.sel_inst_textbox = polish_text_edit(QPlainTextEdit(), 150)
        instruction_card.vbox.addWidget(self.sel_inst_textbox)
        layout.addWidget(instruction_card)

        index_card = Card("Static GM Index", "Browse to your AIInfluence folder. The app indexes editable .txt files, summarizes [GM] ## children, and sends all [GM] child summaries to the selector.")
        idx_grid = QGridLayout()
        idx_grid.setColumnStretch(1, 1)
        self.static_index_enabled_var = QCheckBox("Enabled")
        self.static_index_folder_entry = polish_line_edit(QLineEdit())
        folder_row = QWidget()
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.setSpacing(8)
        folder_layout.addWidget(self.static_index_folder_entry, 1)
        browse_ai_folder = QPushButton("Browse…")
        browse_ai_folder.clicked.connect(lambda: self.choose_folder(self.static_index_folder_entry))
        folder_layout.addWidget(browse_ai_folder)
        self.static_index_db_entry = polish_line_edit(QLineEdit())
        # Static GM Index reindexing is manual-only.
        self.static_index_summary_enabled_var = QCheckBox("Use LLM summaries during reindex")
        self.static_index_summary_model_entry = make_model_combo()
        self.static_index_summary_model_entry.setPlaceholderText("blank = selector model")
        summary_model_row = QWidget()
        summary_model_layout = QHBoxLayout(summary_model_row)
        summary_model_layout.setContentsMargins(0, 0, 0, 0)
        summary_model_layout.setSpacing(8)
        summary_model_layout.addWidget(self.static_index_summary_model_entry, 1)
        self.fetch_summary_models_btn = QPushButton("Fetch Models")
        self.fetch_summary_models_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.fetch_summary_models_btn.setIconSize(QSize(16, 16))
        self.fetch_summary_models_btn.clicked.connect(lambda: self.fetch_models("summary"))
        summary_model_layout.addWidget(self.fetch_summary_models_btn)
        self.static_index_summary_prompt_textbox = polish_text_edit(QPlainTextEdit(), 110)
        self.static_index_summary_prompt_textbox.setPlaceholderText(DEFAULT_SUMMARY_INSTRUCTION)
        self.static_index_summary_prompt_textbox.setToolTip("Prompt used only while reindexing to create summaries for selector matching. Selected IDs still inject the full original content.")
        self.static_index_files_entry = polish_line_edit(QLineEdit())
        self.static_index_files_entry.setToolTip("Comma-separated filenames inside the AIInfluence folder.")
        add_grid_row(idx_grid, 0, "Static GM index", self.static_index_enabled_var)
        add_grid_row(idx_grid, 1, "AIInfluence folder", folder_row)
        add_grid_row(idx_grid, 2, "Indexed files", self.static_index_files_entry)
        add_grid_row(idx_grid, 3, "DB path", self.static_index_db_entry)
        add_grid_row(idx_grid, 4, "LLM summaries", self.static_index_summary_enabled_var)
        add_grid_row(idx_grid, 5, "Summary model override", summary_model_row)
        add_grid_row(idx_grid, 6, "Summary prompt", self.static_index_summary_prompt_textbox)
        index_card.vbox.addLayout(idx_grid)
        index_buttons = QHBoxLayout()
        index_hint = QLabel("Summary model/API defaults to the Selector Model settings, so GLM/GPT can summarize once at reindex and the selector later sees only summaries + IDs.")
        index_hint.setObjectName("Hint")
        index_hint.setWordWrap(True)
        self.reindex_static_btn = QPushButton("Save & Reindex DB")
        self.reindex_static_btn.setObjectName("PrimaryButton")
        self.reindex_static_btn.clicked.connect(self.save_and_reindex_static_db)
        self.signals.reindex_button_enabled.connect(self.reindex_static_btn.setEnabled)
        index_buttons.addWidget(index_hint, 1)
        index_buttons.addWidget(self.reindex_static_btn)
        index_card.vbox.addLayout(index_buttons)
        layout.addWidget(index_card)

        layout.addStretch(1)
        return scroll

    def build_character_memory_tab(self) -> QWidget:
        scroll, page, layout = make_scroll_page()

        intro = Card(
            "Character Memory Control",
            "Compresses AIInfluence save_data character ConversationHistory in the campaign JSON files before AIInfluence builds prompts."
        )
        hint = QLabel("Use this when Conversation History context is too large. Old raw lines become MEMORY1/MEMORY2 entries, while the newest raw lines stay untouched.")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        intro.vbox.addWidget(hint)
        layout.addWidget(intro)

        campaign_card = Card("Campaign Data", "Browse to .../Modules/AIInfluence/save_data/<campaign_id>.")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.char_mem_enabled_var = QCheckBox("Enabled")
        self.char_mem_campaign_entry = polish_line_edit(QLineEdit())
        campaign_row = QWidget()
        campaign_layout = QHBoxLayout(campaign_row)
        campaign_layout.setContentsMargins(0, 0, 0, 0)
        campaign_layout.setSpacing(8)
        campaign_layout.addWidget(self.char_mem_campaign_entry, 1)
        browse_campaign = QPushButton("Browse…")
        browse_campaign.clicked.connect(lambda: self.choose_folder(self.char_mem_campaign_entry))
        campaign_layout.addWidget(browse_campaign)
        add_grid_row(grid, 0, "Character Memory", self.char_mem_enabled_var)
        add_grid_row(grid, 1, "Campaign folder", campaign_row)
        campaign_card.vbox.addLayout(grid)
        layout.addWidget(campaign_card)

        llm_card = Card("Memory LLM", "Blank API/model fields fall back to Selector settings, then Main LLM settings.")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.char_mem_api_url_entry = polish_line_edit(QLineEdit())
        self.char_mem_api_key_entry = polish_line_edit(QLineEdit())
        self.char_mem_api_key_entry.setEchoMode(QLineEdit.EchoMode.Password)
        self.char_mem_model_entry = make_model_combo()
        self.char_mem_model_entry.setPlaceholderText("blank = selector/main model")
        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(8)
        model_layout.addWidget(self.char_mem_model_entry, 1)
        self.fetch_char_mem_models_btn = QPushButton("Fetch Models")
        self.fetch_char_mem_models_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.fetch_char_mem_models_btn.setIconSize(QSize(16, 16))
        self.fetch_char_mem_models_btn.clicked.connect(lambda: self.fetch_models("character_memory"))
        model_layout.addWidget(self.fetch_char_mem_models_btn)
        self.char_mem_temp_entry = polish_line_edit(QLineEdit(format_request_parameter_value("temperature", 0.1)))
        self.char_mem_max_tokens_entry = make_spin("int", 32000, (1, 1000000))
        self.char_mem_timeout_entry = make_spin("float", 180.0, (1.0, 7200.0))
        add_grid_row(grid, 0, "API URL", self.char_mem_api_url_entry)
        add_grid_row(grid, 1, "API Key", self.char_mem_api_key_entry)
        add_grid_row(grid, 2, "Model", model_row)
        add_grid_row(grid, 3, "Temperature", self.char_mem_temp_entry)
        add_grid_row(grid, 4, "Max tokens", self.char_mem_max_tokens_entry)
        add_grid_row(grid, 5, "Timeout seconds", self.char_mem_timeout_entry)
        llm_card.vbox.addLayout(grid)
        layout.addWidget(llm_card)

        behavior_card = Card("Conversation Summarization", "Summarizes old raw ConversationHistory lines and preserves the newest lines exactly.")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.char_mem_preserve_lines_entry = make_spin("int", 20, (0, 10000))
        self.char_mem_auto_enabled_var = QCheckBox("Auto summarize")
        self.char_mem_auto_trigger_entry = make_spin("int", 30, (1, 10000))
        self.char_mem_auto_scan_interval_entry = make_spin("float", 30.0, (5.0, 3600.0))
        self.char_mem_auto_debounce_entry = make_spin("float", 8.0, (0.0, 3600.0))
        self.char_mem_max_memory_entries_entry = make_spin("int", 5, (1, 1000))
        add_grid_row(grid, 0, "Preserve last raw lines", self.char_mem_preserve_lines_entry)
        add_grid_row(grid, 1, "Auto mode", self.char_mem_auto_enabled_var)
        add_grid_row(grid, 2, "Auto trigger raw lines", self.char_mem_auto_trigger_entry)
        add_grid_row(grid, 3, "Auto scan interval seconds", self.char_mem_auto_scan_interval_entry)
        add_grid_row(grid, 4, "Auto debounce seconds", self.char_mem_auto_debounce_entry)
        add_grid_row(grid, 5, "Merge MEMORY entries after", self.char_mem_max_memory_entries_entry)
        behavior_card.vbox.addLayout(grid)
        layout.addWidget(behavior_card)

        prompts_card = Card("Prompts", "Summary prompt creates MEMORY entries. Profile prompt conservatively updates personality/backstory.")
        self.char_mem_summary_prompt_textbox = polish_text_edit(QPlainTextEdit(), 130)
        self.char_mem_summary_prompt_textbox.setPlaceholderText(DEFAULT_CHARACTER_MEMORY_SUMMARY_PROMPT)
        self.char_mem_merge_prompt_textbox = polish_text_edit(QPlainTextEdit(), 130)
        self.char_mem_merge_prompt_textbox.setPlaceholderText(DEFAULT_CHARACTER_MEMORY_MERGE_PROMPT)
        self.char_mem_profile_prompt_textbox = polish_text_edit(QPlainTextEdit(), 160)
        self.char_mem_profile_prompt_textbox.setPlaceholderText(DEFAULT_CHARACTER_MEMORY_PROFILE_PROMPT)
        label = QLabel("Conversation summary prompt")
        label.setObjectName("FieldLabel")
        prompts_card.vbox.addWidget(label)
        prompts_card.vbox.addWidget(self.char_mem_summary_prompt_textbox)
        label = QLabel("Memory merge prompt")
        label.setObjectName("FieldLabel")
        prompts_card.vbox.addWidget(label)
        prompts_card.vbox.addWidget(self.char_mem_merge_prompt_textbox)
        label = QLabel("Personality / backstory update prompt")
        label.setObjectName("FieldLabel")
        prompts_card.vbox.addWidget(label)
        prompts_card.vbox.addWidget(self.char_mem_profile_prompt_textbox)
        layout.addWidget(prompts_card)

        actions_card = Card("Actions", "Back up before destructive operations. Summarize/update calls may take a while for large campaigns.")
        self.char_mem_backup_before_var = QCheckBox("Create backup before summarize/update")
        button_row = QHBoxLayout()
        self.char_mem_scan_btn = QPushButton("Scan Campaign")
        self.char_mem_scan_btn.clicked.connect(self.scan_character_memory)
        self.char_mem_backup_btn = QPushButton("Backup Current Campaign Data")
        self.char_mem_backup_btn.clicked.connect(self.backup_character_memory)
        self.char_mem_summarize_btn = QPushButton("Summarize Conversation History")
        self.char_mem_summarize_btn.setObjectName("PrimaryButton")
        self.char_mem_summarize_btn.clicked.connect(self.summarize_character_memory)
        self.char_mem_update_profile_btn = QPushButton("Update Character Personality / Backstory")
        self.char_mem_update_profile_btn.clicked.connect(self.update_character_profiles)
        for button in (self.char_mem_scan_btn, self.char_mem_backup_btn, self.char_mem_summarize_btn, self.char_mem_update_profile_btn):
            button_row.addWidget(button)
        actions_card.vbox.addWidget(self.char_mem_backup_before_var)
        actions_card.vbox.addLayout(button_row)
        self.char_mem_result_box = polish_text_edit(QPlainTextEdit(), 220)
        self.char_mem_result_box.setReadOnly(True)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.char_mem_result_box.setFont(mono)
        actions_card.vbox.addWidget(self.char_mem_result_box)
        layout.addWidget(actions_card)

        layout.addStretch(1)
        return scroll

    def build_detection_tab(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        header = Card("Request Type Detection", "Strings used to resolve incoming requests as dialogue, diplomacy, or events. Add as many signatures as needed.")
        row = QHBoxLayout()
        self.add_detection_button = QPushButton("Add Detection")
        self.add_detection_button.setObjectName("PrimaryButton")
        self.add_detection_button.clicked.connect(self.add_detection_rule)
        row.addStretch(1)
        row.addWidget(self.add_detection_button)
        header.vbox.addLayout(row)
        layout.addWidget(header)

        self.detection_scroll, self.detection_page, self.detection_layout = make_scroll_page()
        layout.addWidget(self.detection_scroll, 1)
        return root

    def build_log_viewer_tab(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.llm_log_view = self.make_log_pane("LLM Request/Response Log")
        self.selector_log_view = self.make_log_pane("Selector Log")
        splitter.addWidget(self.llm_log_view[0])
        splitter.addWidget(self.selector_log_view[0])
        splitter.setSizes([1, 1])
        layout.addWidget(splitter, 1)
        return root

    def make_log_pane(self, title: str) -> Tuple[QWidget, QPlainTextEdit]:
        pane = QWidget()
        v = QVBoxLayout(pane)
        v.setContentsMargins(0, 0, 0, 0)
        label = QLabel(title)
        label.setObjectName("CardTitle")
        edit = polish_text_edit(QPlainTextEdit(), 420)
        edit.setReadOnly(True)
        edit.setProperty("_gm_follow_tail", True)
        edit.setProperty("_gm_updating_log", False)
        edit.setProperty("_gm_last_log_text", "")
        edit.setProperty("_gm_user_scrolled", False)
        edit.verticalScrollBar().valueChanged.connect(
            lambda value, e=edit: self.on_log_scroll_changed(e, value)
        )
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        edit.setFont(mono)
        v.addWidget(label)
        v.addWidget(edit, 1)
        return pane, edit

    # ------------------------------------------------------------------
    # Rules tabs
    # ------------------------------------------------------------------
    def drop_rule_schema(self) -> List[Tuple[str, str, str]]:
        return [
            ("name", "line", "Rule name"),
            ("request_types", "list", "Request types, comma-separated"),
            ("drop_beginning", "text", "Drop beginning marker"),
            ("drop_end", "text", "Drop end marker"),
            ("delete_drop_beginning_marker", "bool", "Delete beginning marker"),
            ("delete_drop_end_marker", "bool", "Delete end marker"),
        ]

    def replace_rule_schema(self) -> List[Tuple[str, str, str]]:
        return [
            ("name", "line", "Rule name"),
            ("request_types", "list", "Request types, comma-separated"),
            ("replace_beginning", "text", "Replace beginning marker"),
            ("replace_end", "text", "Replace end marker"),
            ("replacement_text", "text_large", "Replacement text"),
            ("delete_replace_beginning_marker", "bool", "Delete beginning marker"),
            ("delete_replace_end_marker", "bool", "Delete end marker"),
        ]

    def context_rule_schema(self) -> List[Tuple[str, str, str]]:
        return [
            ("name", "line", "Rule name"),
            ("request_types", "list", "Request types, comma-separated"),
            ("beginning", "text", "Context beginning marker"),
            ("end", "text", "Context end marker"),
            ("include_beginning_marker", "bool", "Include beginning marker"),
            ("include_end_marker", "bool", "Include end marker"),
            ("limit_enabled", "context_limit", "Limit output"),
        ]

    def build_rules_tab(self, rule_kind: str, title: str, schema: List[Tuple[str, str, str]]) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        header = Card(title, "Marker-based rules are fail-closed: if configured markers are not found, the rule is skipped rather than deleting too much.")
        add = QPushButton("Add Rule")
        add.setObjectName("PrimaryButton")
        add.clicked.connect(lambda _=False, kind=rule_kind: self.add_rule(kind))
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(add)
        header.vbox.addLayout(row)
        layout.addWidget(header)

        scroll, page, list_layout = make_scroll_page()
        setattr(self, f"{rule_kind}_scroll", scroll)
        setattr(self, f"{rule_kind}_page", page)
        setattr(self, f"{rule_kind}_layout", list_layout)
        layout.addWidget(scroll, 1)
        return root

    # ------------------------------------------------------------------
    # Config ⇄ UI
    # ------------------------------------------------------------------
    def apply_config_to_ui(self) -> None:
        cfg = self.settings
        server = cfg.get("server", {})
        self.host_entry.setText(str(server.get("host", "localhost")))
        self.port_entry.setValue(int(server.get("port", 5100)))

        llm = cfg.get("llm", {})
        self.api_url_entry.setText(str(llm.get("api_url", "")))
        self.api_key_entry.setText(str(llm.get("api_key", "")))
        self.site_url_entry.setText(str(llm.get("site_url", "")))
        self.app_title_entry.setText(str(llm.get("app_title", "GameMaster")))
        models = llm.get("models", {}) if isinstance(llm.get("models"), dict) else {}
        for request_type, field in self.model_fields.items():
            set_combo_text(field, str(models.get(request_type, "")))

        self.apply_gm_to_ui()
        self.apply_request_parameters_to_ui()
        self.apply_system_prompts_to_ui()
        self.apply_selector_to_ui()
        self.apply_character_memory_to_ui()
        self.apply_detection_to_ui()
        self.apply_rules_to_ui()
        self.apply_logs_to_ui()

    def apply_gm_to_ui(self) -> None:
        gm = self.get_gm_dict()
        for key, widget in self.gm_widgets.items():
            _, kind, _, default = KNOWN_GM_FIELDS[key]
            value = gm.get(key, default)
            self.set_widget_value(widget, value, kind)

        dynamic_hide = gm.get("dynamic_hide_until_relevant", {})
        if not isinstance(dynamic_hide, dict):
            dynamic_hide = {}
        for key, widget in self.gm_dynamic_hide_widgets.items():
            widget.setChecked(text_to_bool(dynamic_hide.get(key, DYNAMIC_HIDE_UNTIL_RELEVANT_DEFAULTS.get(key, True))))

        # Rebuild editable extras for simple unknown gm keys.
        self.clear_layout(self.gm_extras_layout)
        self.gm_extra_widgets.clear()
        row = 0
        for key, value in sorted(gm.items()):
            if key in KNOWN_GM_FIELDS or key in HIDDEN_GM_FIELDS or key in {"prompt_drop_rules", "prompt_replace_rules", "dynamic_hide_until_relevant"}:
                continue
            if isinstance(value, (bool, int, float, str)) or value is None:
                kind = "bool" if isinstance(value, bool) else "float" if isinstance(value, float) else "int" if isinstance(value, int) else "line"
                widget = self.make_setting_widget(kind, value if value is not None else "", None)
                self.gm_extra_widgets[key] = widget
                add_grid_row(self.gm_extras_layout, row, key, widget)
                row += 1
        self.gm_extras_card.setVisible(row > 0)

    def apply_request_parameters_to_ui(self) -> None:
        params = self.settings.get("request_parameters", {})
        for request_type, widgets in self.request_parameter_widgets.items():
            rt_params = params.get(request_type, {}) if isinstance(params.get(request_type), dict) else {}
            for name, (check, spin) in widgets.items():
                raw = rt_params.get(name, {})
                if isinstance(raw, dict):
                    enabled = text_to_bool(raw.get("enabled", False))
                    value = raw.get("value", 40 if name == "top_k" else 1.0)
                else:
                    enabled = raw is not None
                    value = raw if raw is not None else (40 if name == "top_k" else 1.0)
                check.setChecked(enabled)
                spin.setText(format_request_parameter_value(name, value))
                spin.setEnabled(enabled)

    def apply_system_prompts_to_ui(self) -> None:
        prompts = self.settings.get("system_prompts", {})
        for request_type, widgets in self.system_prompt_widgets.items():
            block = prompts.get(request_type, {}) if isinstance(prompts.get(request_type), dict) else {}
            widgets["pre_history"].setPlainText(str(block.get("pre_history", "") or ""))
            widgets["post_history"].setPlainText(str(block.get("post_history", "") or ""))

    def apply_selector_to_ui(self) -> None:
        selector = self.settings.get("selector", {})
        self.sel_api_url_entry.setText(str(selector.get("api_url", "")))
        self.sel_api_key_entry.setText(str(selector.get("api_key", "")))
        set_combo_text(self.sel_model_entry, str(selector.get("model", "")))
        self.sel_temp_entry.setText(format_request_parameter_value("temperature", selector.get("temperature", 0.0)))
        self.sel_max_tokens_entry.setValue(int(selector.get("max_tokens", 32000) or 32000))
        self.sel_timeout_entry.setValue(float(selector.get("timeout_seconds", 120.0) or 120.0))
        self.sel_inst_textbox.setPlainText(str(selector.get("instruction", DEFAULT_SELECTOR_INSTRUCTION) or ""))
        self.sel_log_enabled_var.setChecked(text_to_bool(selector.get("log_enabled", False)))
        self.sel_log_path_entry.setText(str(selector.get("log_path", "logs/selector-log.txt")))
        self.sel_log_pretty_var.setChecked(text_to_bool(selector.get("pretty_json", True)))

        static_index = self.settings.get("static_gm_index", {}) if isinstance(self.settings.get("static_gm_index"), dict) else {}
        self.static_index_enabled_var.setChecked(text_to_bool(static_index.get("enabled", True)))
        self.static_index_folder_entry.setText(str(static_index.get("ai_influence_folder", "") or ""))
        files = static_index.get("files", DEFAULT_STATIC_GM_INDEX_FILES)
        if isinstance(files, list):
            self.static_index_files_entry.setText(", ".join(str(x) for x in files))
        else:
            self.static_index_files_entry.setText(str(files or ", ".join(DEFAULT_STATIC_GM_INDEX_FILES)))
        self.static_index_db_entry.setText(str(static_index.get("db_path", "cache/static_gm_index.sqlite3") or "cache/static_gm_index.sqlite3"))
        # auto_reindex is deprecated/ignored; reindexing happens only via Save & Reindex DB.
        self.static_index_summary_enabled_var.setChecked(text_to_bool(static_index.get("summary_enabled", True)))
        set_combo_text(self.static_index_summary_model_entry, str(static_index.get("summary_model", "") or ""))
        self.static_index_summary_prompt_textbox.setPlainText(
            str(static_index.get("summary_instruction", "") or DEFAULT_SUMMARY_INSTRUCTION)
        )


    def apply_character_memory_to_ui(self) -> None:
        cm = self.settings.get("character_memory", {}) if isinstance(self.settings.get("character_memory"), dict) else {}
        self.char_mem_enabled_var.setChecked(text_to_bool(cm.get("enabled", True)))
        self.char_mem_campaign_entry.setText(str(cm.get("campaign_dir", "") or ""))
        self.char_mem_api_url_entry.setText(str(cm.get("api_url", "") or ""))
        self.char_mem_api_key_entry.setText(str(cm.get("api_key", "") or ""))
        set_combo_text(self.char_mem_model_entry, str(cm.get("model", "") or ""))
        self.char_mem_temp_entry.setText(format_request_parameter_value("temperature", cm.get("temperature", 0.1)))
        self.char_mem_max_tokens_entry.setValue(int(cm.get("max_tokens", 32000) or 32000))
        self.char_mem_timeout_entry.setValue(float(cm.get("timeout_seconds", 180.0) or 180.0))
        self.char_mem_preserve_lines_entry.setValue(int(cm.get("preserve_last_lines", 20) or 20))
        self.char_mem_auto_enabled_var.setChecked(text_to_bool(cm.get("auto_enabled", False)))
        self.char_mem_auto_trigger_entry.setValue(int(cm.get("auto_trigger_raw_lines", 30) or 30))
        self.char_mem_auto_scan_interval_entry.setValue(float(cm.get("auto_scan_interval_seconds", 30.0) or 30.0))
        self.char_mem_auto_debounce_entry.setValue(float(cm.get("auto_debounce_seconds", 8.0) or 8.0))
        self.char_mem_max_memory_entries_entry.setValue(int(cm.get("max_memory_entries", 5) or 5))
        self.char_mem_summary_prompt_textbox.setPlainText(str(cm.get("summary_prompt", "") or DEFAULT_CHARACTER_MEMORY_SUMMARY_PROMPT))
        self.char_mem_merge_prompt_textbox.setPlainText(str(cm.get("merge_prompt", "") or DEFAULT_CHARACTER_MEMORY_MERGE_PROMPT))
        self.char_mem_profile_prompt_textbox.setPlainText(str(cm.get("profile_update_prompt", "") or DEFAULT_CHARACTER_MEMORY_PROFILE_PROMPT))

    def apply_detection_to_ui(self) -> None:
        self.detection_rules = []
        detection = self.settings.get("request_type_detection", {})
        if isinstance(detection, dict):
            for request_type, strings in detection.items():
                if isinstance(strings, list):
                    for text in strings:
                        self.detection_rules.append({"type": request_type, "string": str(text)})
                elif isinstance(strings, str):
                    self.detection_rules.append({"type": request_type, "string": strings})
        self.render_detection_rules()

    def apply_rules_to_ui(self) -> None:
        self.drop_rules = deepcopy(self.settings.get("prompt_drop_rules", [])) if isinstance(self.settings.get("prompt_drop_rules"), list) else []
        self.replace_rules = deepcopy(self.settings.get("prompt_replace_rules", [])) if isinstance(self.settings.get("prompt_replace_rules"), list) else []
        selector = self.settings.get("selector", {})
        self.context_rules = deepcopy(selector.get("context_rules", [])) if isinstance(selector.get("context_rules"), list) else []
        for rule in [*self.drop_rules, *self.replace_rules, *self.context_rules]:
            self.strip_obsolete_rule_fields(rule)
        self.render_rules("drop")
        self.render_rules("replace")
        self.render_rules("context")

    def apply_logs_to_ui(self) -> None:
        gui = self.settings.get("gui", {})
        llm_logging = self.settings.get("llm_logging", {})
        selector = self.settings.get("selector", {})
        self.log_viewer_enabled_var.setChecked(text_to_bool(gui.get("log_viewer_enabled", False)))
        self.llm_log_enabled_var.setChecked(text_to_bool(llm_logging.get("enabled", False)))
        self.llm_log_path_entry.setText(str(llm_logging.get("path", "logs/llm-log.txt")))
        self.llm_log_pretty_var.setChecked(text_to_bool(llm_logging.get("pretty_json", True)))
        self.sel_log_path_entry.setText(str(selector.get("log_path", "logs/selector-log.txt")))
        self.on_log_viewer_toggled(self.log_viewer_enabled_var.isChecked())

    def collect_config_from_ui(self) -> Dict[str, Any]:
        cfg = merge_defaults(self.settings)
        cfg["server"] = {
            "host": self.host_entry.text().strip() or "localhost",
            "port": int(self.port_entry.value()),
        }
        cfg.setdefault("llm", {})
        cfg["llm"]["api_url"] = self.api_url_entry.text().strip()
        cfg["llm"]["api_key"] = self.api_key_entry.text()
        cfg["llm"]["site_url"] = self.site_url_entry.text().strip()
        cfg["llm"]["app_title"] = self.app_title_entry.text().strip() or "GameMaster"
        cfg["llm"]["models"] = {request_type: combo_text(field) for request_type, field in self.model_fields.items()}

        gm = self.collect_gm_from_ui()
        cfg["gm"] = deepcopy(gm)
        cfg.pop("retr" + "ieval", None)

        cfg["request_parameters"] = self.collect_request_parameters_from_ui()
        cfg["system_prompts"] = self.collect_system_prompts_from_ui()

        cfg["prompt_drop_rules"] = self.collect_rules_from_ui("drop")
        cfg["prompt_replace_rules"] = self.collect_rules_from_ui("replace")
        cfg["gm"].pop("prompt_drop_rules", None)
        cfg["gm"].pop("prompt_replace_rules", None)

        cfg["request_type_detection"] = self.collect_detection_from_ui()

        filtering = cfg.setdefault("filtering", {})
        filtering["mode"] = DEFAULT_FILTERING_MODE

        selector = cfg.setdefault("selector", {})
        selector["enabled"] = True
        selector["api_url"] = self.sel_api_url_entry.text().strip()
        selector["api_key"] = self.sel_api_key_entry.text()
        selector["model"] = combo_text(self.sel_model_entry)
        try:
            selector["temperature"] = float(parse_request_parameter_value("temperature", self.sel_temp_entry.text()))
        except Exception:
            selector["temperature"] = 0.0
        self.sel_temp_entry.setText(format_request_parameter_value("temperature", selector["temperature"]))
        selector["max_tokens"] = int(self.sel_max_tokens_entry.value())
        selector["timeout_seconds"] = float(self.sel_timeout_entry.value())
        selector["instruction"] = plain_text(self.sel_inst_textbox)
        selector["log_enabled"] = bool(self.sel_log_enabled_var.isChecked())
        # Selector log path now lives on the General tab.
        selector["log_path"] = self.sel_log_path_entry.text().strip() or "logs/selector-log.txt"
        selector["pretty_json"] = bool(self.sel_log_pretty_var.isChecked())
        selector["context_rules"] = self.collect_rules_from_ui("context")

        static_index = cfg.setdefault("static_gm_index", {})
        static_index["enabled"] = bool(self.static_index_enabled_var.isChecked())
        static_index["ai_influence_folder"] = self.static_index_folder_entry.text().strip()
        static_index["files"] = [x.strip() for x in self.static_index_files_entry.text().split(",") if x.strip()] or list(DEFAULT_STATIC_GM_INDEX_FILES)
        static_index["db_path"] = self.static_index_db_entry.text().strip() or "cache/static_gm_index.sqlite3"
        static_index.pop("auto_reindex", None)
        static_index.pop("max_candidates", None)
        static_index.pop("min" + "_similarity", None)
        static_index["selector_payload"] = "summary"
        static_index["summary_enabled"] = bool(self.static_index_summary_enabled_var.isChecked())
        static_index["summary_model"] = combo_text(self.static_index_summary_model_entry)
        static_index.setdefault("summary_api_url", "")
        static_index.setdefault("summary_api_key", "")
        static_index.setdefault("summary_temperature", 0.1)
        static_index.setdefault("summary_max_tokens", 220)
        static_index.setdefault("summary_timeout_seconds", 120.0)
        static_index.setdefault("summary_max_chars", 6000)
        static_index["summary_instruction"] = (
            plain_text(self.static_index_summary_prompt_textbox).strip() or DEFAULT_SUMMARY_INSTRUCTION
        )


        character_memory = cfg.setdefault("character_memory", {})
        character_memory["enabled"] = bool(self.char_mem_enabled_var.isChecked())
        character_memory["campaign_dir"] = self.char_mem_campaign_entry.text().strip()
        character_memory["api_url"] = self.char_mem_api_url_entry.text().strip()
        character_memory["api_key"] = self.char_mem_api_key_entry.text()
        character_memory["model"] = combo_text(self.char_mem_model_entry)
        try:
            character_memory["temperature"] = float(parse_request_parameter_value("temperature", self.char_mem_temp_entry.text()))
        except Exception:
            character_memory["temperature"] = 0.1
        self.char_mem_temp_entry.setText(format_request_parameter_value("temperature", character_memory["temperature"]))
        character_memory["max_tokens"] = int(self.char_mem_max_tokens_entry.value())
        character_memory["timeout_seconds"] = float(self.char_mem_timeout_entry.value())
        character_memory["preserve_last_lines"] = int(self.char_mem_preserve_lines_entry.value())
        character_memory["auto_enabled"] = bool(self.char_mem_auto_enabled_var.isChecked())
        character_memory["auto_trigger_raw_lines"] = int(self.char_mem_auto_trigger_entry.value())
        character_memory["auto_scan_interval_seconds"] = float(self.char_mem_auto_scan_interval_entry.value())
        character_memory["auto_debounce_seconds"] = float(self.char_mem_auto_debounce_entry.value())
        character_memory["max_memory_entries"] = int(self.char_mem_max_memory_entries_entry.value())
        character_memory["summary_prompt"] = plain_text(self.char_mem_summary_prompt_textbox).strip() or DEFAULT_CHARACTER_MEMORY_SUMMARY_PROMPT
        character_memory["merge_prompt"] = plain_text(self.char_mem_merge_prompt_textbox).strip() or DEFAULT_CHARACTER_MEMORY_MERGE_PROMPT
        character_memory["profile_update_prompt"] = plain_text(self.char_mem_profile_prompt_textbox).strip() or DEFAULT_CHARACTER_MEMORY_PROFILE_PROMPT

        cfg["llm_logging"] = {
            "enabled": bool(self.llm_log_enabled_var.isChecked()),
            "path": self.llm_log_path_entry.text().strip() or "logs/llm-log.txt",
            "pretty_json": bool(self.llm_log_pretty_var.isChecked()),
        }
        cfg.setdefault("gui", {})
        cfg["gui"]["log_viewer_enabled"] = bool(self.log_viewer_enabled_var.isChecked())
        return cfg

    # ------------------------------------------------------------------
    # Rendering dynamic lists
    # ------------------------------------------------------------------
    def render_detection_rules(self) -> None:
        self.clear_layout(self.detection_layout)
        self.detection_widgets = []
        for idx, rule in enumerate(self.detection_rules):
            card = Card(f"Detection {idx + 1}")
            header = QHBoxLayout()
            request_type = polish_line_edit(QLineEdit(str(rule.get("type", "dialogue"))))
            delete = QPushButton("Delete")
            delete.setObjectName("DangerButton")
            delete.clicked.connect(lambda _=False, i=idx: self.delete_detection_rule(i))
            request_type.textChanged.connect(lambda text, r=rule: r.__setitem__("type", text))
            header.addWidget(QLabel("Request type"))
            header.addWidget(request_type, 1)
            header.addWidget(delete)
            card.vbox.addLayout(header)
            string_edit = polish_text_edit(QPlainTextEdit(str(rule.get("string", ""))), 95)
            string_edit.textChanged.connect(lambda r=rule, edit=string_edit: r.__setitem__("string", plain_text(edit)))
            card.vbox.addWidget(QLabel("Detection string"))
            card.vbox.addWidget(string_edit)
            self.detection_widgets.append({"type": request_type, "string": string_edit})
            self.detection_layout.addWidget(card)
        self.detection_layout.addStretch(1)

    def add_detection_rule(self) -> None:
        self.collect_detection_from_ui(update_only=True)
        self.detection_rules.append({"type": "dialogue", "string": ""})
        self.render_detection_rules()

    def delete_detection_rule(self, index: int) -> None:
        self.collect_detection_from_ui(update_only=True)
        if 0 <= index < len(self.detection_rules):
            self.detection_rules.pop(index)
        self.render_detection_rules()

    def render_rules(self, rule_kind: str) -> None:
        list_layout: QVBoxLayout = getattr(self, f"{rule_kind}_layout")
        self.clear_layout(list_layout)
        rules = self.rule_list(rule_kind)
        schema = self.rule_schema(rule_kind)
        widgets: List[Dict[str, QWidget]] = []
        for idx, rule in enumerate(rules):
            name = str(rule.get("name", f"Rule {idx + 1}"))
            card = Card(f"Rule {idx + 1}")
            header = QHBoxLayout()
            name_edit = polish_line_edit(QLineEdit(name))
            name_edit.textChanged.connect(lambda text, r=rule: r.__setitem__("name", text))
            delete = QPushButton("Delete")
            delete.setObjectName("DangerButton")
            delete.clicked.connect(lambda _=False, kind=rule_kind, i=idx: self.delete_rule(kind, i))
            header.addWidget(QLabel("Name"))
            header.addWidget(name_edit, 1)
            header.addWidget(delete)
            card.vbox.addLayout(header)

            rule_widgets: Dict[str, QWidget] = {"name": name_edit}
            grid = QGridLayout()
            grid.setColumnStretch(1, 1)
            row = 0
            for key, field_type, label in schema:
                if key == "name":
                    continue
                value = self.rule_value(rule, key)
                if key in PROMPT_RULE_TEXT_KEYS and isinstance(value, str):
                    value = normalize_prompt_rule_text(value)
                    rule[key] = value
                if field_type == "context_limit":
                    enabled = QCheckBox("Preserve only")
                    enabled.setChecked(text_to_bool(rule.get("limit_enabled", False)))
                    try:
                        limit_chars = max(1, int(rule.get("limit_chars", 5000) or 5000))
                    except Exception:
                        limit_chars = 5000
                    chars = make_spin("int", limit_chars, (1, 1000000))
                    position = QComboBox()
                    for position_label, position_value in CONTEXT_LIMIT_POSITIONS:
                        position.addItem(position_label, position_value)
                    current_position = str(rule.get("limit_position", "end") or "end").strip().lower()
                    current_position = current_position.replace(" ", "_").replace("-", "_")
                    if current_position in {"start", "front"}:
                        current_position = "beginning"
                    elif current_position in {"both", "start_and_end"}:
                        current_position = "beginning_and_end"
                    position_index = next(
                        (i for i in range(position.count()) if position.itemData(i) == current_position),
                        1,
                    )
                    position.setCurrentIndex(position_index)

                    row_widget = QWidget()
                    row_layout = QHBoxLayout(row_widget)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    row_layout.setSpacing(8)
                    row_layout.addWidget(enabled)
                    row_layout.addWidget(chars)
                    row_layout.addWidget(QLabel("characters from the"))
                    row_layout.addWidget(position)
                    row_layout.addWidget(QLabel("of extracted context"))
                    row_layout.addStretch(1)

                    def set_context_limit_enabled(
                        checked: bool,
                        *,
                        r: Dict[str, Any] = rule,
                        c: QSpinBox = chars,
                        p: QComboBox = position,
                    ) -> None:
                        r["limit_enabled"] = bool(checked)
                        c.setEnabled(bool(checked))
                        p.setEnabled(bool(checked))

                    enabled.toggled.connect(set_context_limit_enabled)
                    chars.valueChanged.connect(lambda value, r=rule: r.__setitem__("limit_chars", int(value)))
                    position.currentIndexChanged.connect(
                        lambda _idx, r=rule, combo=position: r.__setitem__("limit_position", str(combo.currentData() or "end"))
                    )
                    set_context_limit_enabled(enabled.isChecked())
                    rule["limit_chars"] = int(chars.value())
                    rule["limit_position"] = str(position.currentData() or "end")

                    rule_widgets["limit_enabled"] = enabled
                    rule_widgets["limit_chars"] = chars
                    rule_widgets["limit_position"] = position
                    add_grid_row(grid, row, label, row_widget)
                    row += 1
                    continue
                if field_type == "bool":
                    widget = QCheckBox()
                    widget.setChecked(text_to_bool(value if value is not None else True))
                    widget.toggled.connect(lambda checked, r=rule, k=key: self.set_rule_bool(r, k, checked))
                elif field_type == "list":
                    text = ", ".join(value) if isinstance(value, list) else str(value or "")
                    widget = polish_line_edit(QLineEdit(text))
                    widget.textChanged.connect(lambda text, r=rule, k=key: r.__setitem__(k, [x.strip() for x in text.split(",") if x.strip()]))
                elif field_type == "text_large":
                    widget = polish_text_edit(QPlainTextEdit(str(value or "")), 135)
                    widget.textChanged.connect(lambda r=rule, k=key, edit=widget: r.__setitem__(k, plain_text(edit)))
                elif field_type == "text":
                    widget = polish_text_edit(QPlainTextEdit(str(value or "")), 76)
                    widget.textChanged.connect(lambda r=rule, k=key, edit=widget: r.__setitem__(k, plain_text(edit)))
                else:
                    widget = polish_line_edit(QLineEdit(str(value or "")))
                    widget.textChanged.connect(lambda text, r=rule, k=key: r.__setitem__(k, text))
                rule_widgets[key] = widget
                add_grid_row(grid, row, label, widget)
                row += 1
            card.vbox.addLayout(grid)
            widgets.append(rule_widgets)
            list_layout.addWidget(card)
        list_layout.addStretch(1)
        self.rule_widgets[rule_kind] = widgets

    def add_rule(self, rule_kind: str) -> None:
        self.collect_rules_from_ui(rule_kind, update_only=True)
        rules = self.rule_list(rule_kind)
        schema = self.rule_schema(rule_kind)
        new_rule: Dict[str, Any] = {"name": f"New Rule {len(rules) + 1}"}
        for key, field_type, _ in schema:
            if key == "name":
                continue
            if field_type == "bool":
                new_rule[key] = True
            elif field_type == "list":
                new_rule[key] = []
            elif field_type == "context_limit":
                new_rule["limit_enabled"] = False
                new_rule["limit_chars"] = 5000
                new_rule["limit_position"] = "end"
            else:
                new_rule[key] = ""
        rules.append(new_rule)
        self.render_rules(rule_kind)

    def delete_rule(self, rule_kind: str, index: int) -> None:
        self.collect_rules_from_ui(rule_kind, update_only=True)
        rules = self.rule_list(rule_kind)
        if 0 <= index < len(rules):
            rules.pop(index)
        self.render_rules(rule_kind)

    # ------------------------------------------------------------------
    # Dynamic data collectors
    # ------------------------------------------------------------------
    def collect_gm_from_ui(self) -> Dict[str, Any]:
        gm = deepcopy(self.get_gm_dict())
        # Drop obsolete pre-summary-selector keys. They are not read by the runtime
        # anymore and should not be preserved just because an old settings.json had them.
        for obsolete_key in HIDDEN_GM_FIELDS:
            gm.pop(obsolete_key, None)
        gm.pop("prompt_drop_rules", None)
        gm.pop("prompt_replace_rules", None)
        for key, widget in self.gm_widgets.items():
            _, kind, _, _ = KNOWN_GM_FIELDS[key]
            gm[key] = self.widget_value(widget, kind)
        gm["dynamic_hide_until_relevant"] = {
            key: bool(widget.isChecked())
            for key, widget in self.gm_dynamic_hide_widgets.items()
        }
        for key, widget in self.gm_extra_widgets.items():
            if key in HIDDEN_GM_FIELDS:
                continue
            old_value = gm.get(key)
            kind = "bool" if isinstance(old_value, bool) else "float" if isinstance(old_value, float) else "int" if isinstance(old_value, int) else "line"
            gm[key] = self.widget_value(widget, kind)
        return gm

    def collect_request_parameters_from_ui(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for request_type, widgets in self.request_parameter_widgets.items():
            out[request_type] = {}
            for name, (check, spin) in widgets.items():
                try:
                    value: int | float = parse_request_parameter_value(name, spin.text())
                except Exception:
                    value = 40 if name == "top_k" else 1.0
                spin.setText(format_request_parameter_value(name, value))
                out[request_type][name] = {"enabled": bool(check.isChecked()), "value": value}
        return out

    def collect_system_prompts_from_ui(self) -> Dict[str, Dict[str, str]]:
        out: Dict[str, Dict[str, str]] = {}
        for request_type, widgets in self.system_prompt_widgets.items():
            out[request_type] = {
                "pre_history": plain_text(widgets["pre_history"]),
                "post_history": plain_text(widgets["post_history"]),
            }
        return out

    def collect_detection_from_ui(self, update_only: bool = False) -> Dict[str, List[str]]:
        for idx, widgets in enumerate(self.detection_widgets):
            if idx < len(self.detection_rules):
                self.detection_rules[idx]["type"] = widgets["type"].text().strip()  # type: ignore[attr-defined]
                self.detection_rules[idx]["string"] = plain_text(widgets["string"])  # type: ignore[arg-type]
        if update_only:
            return {}
        out: Dict[str, List[str]] = {}
        for rule in self.detection_rules:
            request_type = str(rule.get("type", "")).strip()
            string = str(rule.get("string", "")).strip()
            if request_type and string:
                out.setdefault(request_type, []).append(string)
        return out

    def collect_rules_from_ui(self, rule_kind: str, update_only: bool = False) -> List[Dict[str, Any]]:
        rules = self.rule_list(rule_kind)
        widgets_list = self.rule_widgets.get(rule_kind, [])
        schema = self.rule_schema(rule_kind)
        for idx, widgets in enumerate(widgets_list):
            if idx >= len(rules):
                continue
            rule = rules[idx]
            self.strip_obsolete_rule_fields(rule)
            for key, field_type, _ in schema:
                widget = widgets.get(key)
                if widget is None:
                    continue
                if field_type == "context_limit":
                    enabled_widget = widgets.get("limit_enabled")
                    chars_widget = widgets.get("limit_chars")
                    position_widget = widgets.get("limit_position")
                    rule["limit_enabled"] = bool(enabled_widget.isChecked()) if isinstance(enabled_widget, QCheckBox) else False
                    rule["limit_chars"] = int(chars_widget.value()) if isinstance(chars_widget, QSpinBox) else 5000
                    rule["limit_position"] = (
                        str(position_widget.currentData() or "end")
                        if isinstance(position_widget, QComboBox)
                        else "end"
                    )
                elif field_type == "bool":
                    self.set_rule_bool(rule, key, bool(widget.isChecked()))  # type: ignore[attr-defined]
                elif field_type == "list":
                    text = widget.text()  # type: ignore[attr-defined]
                    rule[key] = [x.strip() for x in text.split(",") if x.strip()]
                elif field_type in {"text", "text_large"}:
                    value = plain_text(widget)  # type: ignore[arg-type]
                    if key in PROMPT_RULE_TEXT_KEYS:
                        value = normalize_prompt_rule_text(value)
                    rule[key] = value
                else:
                    rule[key] = widget.text()  # type: ignore[attr-defined]
        for rule in rules:
            self.strip_obsolete_rule_fields(rule)
        return [] if update_only else deepcopy(rules)


    # ------------------------------------------------------------------
    # Model fetching
    # ------------------------------------------------------------------
    def fetch_models(self, target: str) -> None:
        if target == "selector":
            api_url = self.sel_api_url_entry.text().strip() or self.api_url_entry.text().strip()
            api_key = self.sel_api_key_entry.text() or self.api_key_entry.text()
            site_url = self.site_url_entry.text().strip()
            app_title = self.app_title_entry.text().strip() or "GameMaster"
            button = self.fetch_selector_models_btn
        elif target == "summary":
            static_index = self.settings.get("static_gm_index", {}) if isinstance(self.settings.get("static_gm_index"), dict) else {}
            api_url = str(static_index.get("summary_api_url") or "").strip() or self.sel_api_url_entry.text().strip() or self.api_url_entry.text().strip()
            api_key = str(static_index.get("summary_api_key") or "") or self.sel_api_key_entry.text() or self.api_key_entry.text()
            site_url = self.site_url_entry.text().strip()
            app_title = self.app_title_entry.text().strip() or "GameMaster"
            button = self.fetch_summary_models_btn
        elif target == "character_memory":
            api_url = self.char_mem_api_url_entry.text().strip() or self.sel_api_url_entry.text().strip() or self.api_url_entry.text().strip()
            api_key = self.char_mem_api_key_entry.text() or self.sel_api_key_entry.text() or self.api_key_entry.text()
            site_url = self.site_url_entry.text().strip()
            app_title = self.app_title_entry.text().strip() or "GameMaster"
            button = self.fetch_char_mem_models_btn
        else:
            api_url = self.api_url_entry.text().strip()
            api_key = self.api_key_entry.text()
            site_url = self.site_url_entry.text().strip()
            app_title = self.app_title_entry.text().strip() or "GameMaster"
            button = self.fetch_main_models_btn
        try:
            url = normalize_models_url(api_url)
        except Exception as exc:
            QMessageBox.warning(self, "Fetch models", str(exc))
            return
        button.setEnabled(False)
        self.set_status("Fetching models…", "info")

        def worker() -> None:
            try:
                headers = {
                    "Accept": "application/json",
                    "User-Agent": "GameMaster-GM-GUI/1.0",
                }
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                if site_url:
                    headers["HTTP-Referer"] = site_url
                if app_title:
                    headers["X-Title"] = app_title
                req = request.Request(url, headers=headers, method="GET")
                with request.urlopen(req, timeout=25) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                payload = json.loads(raw)
                models = parse_model_ids(payload)
                if not models:
                    raise RuntimeError("The endpoint responded, but no model IDs were found in the response.")
                self.signals.models_fetched.emit(target, models, "")
            except Exception as exc:
                self.signals.models_fetched.emit(target, [], str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def apply_fetched_models(self, target: str, models: List[str], error: str) -> None:
        if target == "selector":
            self.fetch_selector_models_btn.setEnabled(True)
            combos = [self.sel_model_entry]
        elif target == "summary":
            self.fetch_summary_models_btn.setEnabled(True)
            combos = [self.static_index_summary_model_entry]
        elif target == "character_memory":
            self.fetch_char_mem_models_btn.setEnabled(True)
            combos = [self.char_mem_model_entry]
        else:
            self.fetch_main_models_btn.setEnabled(True)
            combos = list(self.model_fields.values())
        if error:
            self.set_status("Fetch failed", "danger")
            QMessageBox.warning(self, "Fetch models failed", error)
            return
        for combo in combos:
            current = combo_text(combo)
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(models)
            if current:
                set_combo_text(combo, current)
            combo.blockSignals(False)
        self.set_status(f"Fetched {len(models)} models", "success")

    # ------------------------------------------------------------------
    # IO / server actions
    # ------------------------------------------------------------------
    def choose_settings_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open settings.json", str(self.settings_path), "JSON files (*.json);;All files (*.*)")
        if not path:
            return
        self.settings_path = Path(path)
        self.settings = load_json_settings(self.settings_path)
        self.apply_config_to_ui()
        self.set_status(f"Loaded {self.settings_path.name}", "info")

    def choose_log_file(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose log file", str(path_from_text(target.text() or "logs")), "Log/Text files (*.txt *.log *.jsonl);;All files (*.*)")
        if path:
            try:
                p = Path(path)
                rel = p.relative_to(APP_ROOT)
                target.setText(str(rel).replace("\\", "/"))
            except Exception:
                target.setText(path)

    def choose_folder(self, target: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose folder", str(path_from_text(target.text() or ".")))
        if path:
            try:
                p = Path(path)
                rel = p.relative_to(APP_ROOT)
                target.setText(str(rel).replace("\\", "/"))
            except Exception:
                target.setText(path)

    def save_settings(self, and_apply: bool = True) -> None:
        old_host = self.settings.get("server", {}).get("host", "localhost")
        old_port = self.settings.get("server", {}).get("port", 5100)
        self.settings = self.collect_config_from_ui()
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(json.dumps(self.settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            self.set_status("Saved", "success")
            if and_apply:
                server = self.settings.get("server", {})
                if str(old_host) != str(server.get("host")) or str(old_port) != str(server.get("port")):
                    self.restart_server()
                else:
                    self.reload_server()
            self.on_log_viewer_toggled(self.log_viewer_enabled_var.isChecked())
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save settings:\n{exc}")

    def start_server(self) -> None:
        if self.server_process is not None and self.server_process.poll() is None:
            return
        filtering = self.settings.get("filtering", {}) if isinstance(self.settings.get("filtering"), dict) else {}
        filtering_mode = str(filtering.get("mode", DEFAULT_FILTERING_MODE) or DEFAULT_FILTERING_MODE).strip().lower()
        selector = self.settings.get("selector", {}) if isinstance(self.settings.get("selector"), dict) else {}
        if filtering_mode == "selector" and (not selector.get("api_url") or not selector.get("model")):
            self.set_status("Setup required", "warning")
            return

        try:
            env = os.environ.copy()
            env.setdefault("PYTHONUTF8", "1")
            # The GUI-managed server should use the selected settings.json as the source
            # of truth. Stale shell variables like GMR_SELECTOR_MODEL/GMR_MODEL can make
            # hot reload appear broken by overriding saved GUI fields.
            for key in list(env.keys()):
                if key.startswith("GMR_SELECTOR_") or key == "GMR_MODEL" or key.startswith("GMR_MODEL_"):
                    env.pop(key, None)
            env["GMR_CONFIG_PATH"] = str(self.settings_path)
            if getattr(sys, "frozen", False):
                # In --onefile builds there is no external run.py. Relaunch the
                # same EXE in server mode so the GUI can still stop/restart it.
                cmd = [sys.executable, "--server"]
            else:
                script = APP_ROOT / "run.py"
                if not script.exists():
                    self.set_status("run.py missing", "danger")
                    return
                python_exe = sys.executable
                venv_python = APP_ROOT / "venv" / "Scripts" / "python.exe"
                if venv_python.exists():
                    python_exe = str(venv_python)
                cmd = [python_exe, str(script)]
            self.server_process = subprocess.Popen(cmd, cwd=str(APP_ROOT), env=env)
            self.set_status("Running", "success")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        except Exception as exc:
            QMessageBox.critical(self, "Start failed", f"Could not start server:\n{exc}")
            self.set_status("Start failed", "danger")

    def stop_server(self) -> None:
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except Exception:
                try:
                    self.server_process.kill()
                except Exception:
                    pass
            self.server_process = None
        self.set_status("Stopped", "danger")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def restart_server(self) -> None:
        self.stop_server()
        time.sleep(0.4)
        self.start_server()

    def reload_server(self) -> None:
        server = self.settings.get("server", {})
        url = f"http://{server.get('host', 'localhost')}:{server.get('port', 5100)}/reload"

        def do_request() -> None:
            try:
                data = json.dumps({"config_path": str(self.settings_path)}).encode("utf-8")
                req = request.Request(
                    url,
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "X-GameMaster-Config-Path": str(self.settings_path),
                    },
                    method="POST",
                )
                with request.urlopen(req, timeout=20) as response:
                    body = response.read().decode("utf-8", errors="replace")
                    if response.status == 200:
                        try:
                            result = json.loads(body) if body else {}
                        except Exception:
                            result = {}
                        model = str(result.get("selector_model", "") or "").strip() if isinstance(result, dict) else ""
                        temp = result.get("selector_temperature", None) if isinstance(result, dict) else None
                        if model:
                            self.set_status_threadsafe(f"Hot-reloaded selector: {model} temp={temp}", "info")
                        else:
                            self.set_status_threadsafe("Hot-reloaded", "info")
                    else:
                        print(f"Hot-reload failed: HTTP {response.status} {body}")
                        self.set_status_threadsafe(f"Reload failed: HTTP {response.status}", "warning")
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                print(f"Hot-reload failed: HTTP {exc.code} {body}")
                self.set_status_threadsafe(f"Reload failed: HTTP {exc.code}", "warning")
            except Exception as exc:
                # The GUI may not own the server process, so always try the HTTP reload first.
                # If nothing is listening and the GUI owns no running server, start one as a fallback.
                print(f"Hot-reload failed: {exc}")
                if self.server_process is None or self.server_process.poll() is not None:
                    self.set_status_threadsafe("Starting server", "info")
                    self.signals.start_server_requested.emit()
                else:
                    self.set_status_threadsafe("Reload failed", "warning")

        threading.Thread(target=do_request, daemon=True).start()

    def save_and_reindex_static_db(self) -> None:
        # /reindex reloads settings itself, so do not also trigger /reload here.
        # Running both requests at once can leave the header status stuck on "Reindexing DB…"
        # or overwritten by a late hot-reload status.
        self.save_settings(and_apply=False)
        server = self.settings.get("server", {})
        url = f"http://{server.get('host', 'localhost')}:{server.get('port', 5100)}/reindex"
        self.reindex_static_btn.setEnabled(False)
        self.set_status("Reindexing DB…", "info")

        def do_request() -> None:
            try:
                data = json.dumps({"config_path": str(self.settings_path)}).encode("utf-8")
                req = request.Request(
                    url,
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "X-GameMaster-Config-Path": str(self.settings_path),
                    },
                    method="POST",
                )
                with request.urlopen(req, timeout=600) as response:
                    body = response.read().decode("utf-8", errors="replace")
                    payload = json.loads(body) if body else {}
                stats = payload.get("stats", {}) if isinstance(payload, dict) else {}
                static_stats = stats.get("static_gm_index", {}) if isinstance(stats, dict) else {}
                count = static_stats.get("elements", static_stats.get("gm_child_elements", "?"))
                self.set_status_threadsafe(f"Reindexed {count} GM elements", "success")
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                print(f"Reindex failed: HTTP {exc.code} {body}")
                self.set_status_threadsafe(f"Reindex failed: HTTP {exc.code}", "warning")
            except Exception as exc:
                print(f"Reindex failed: {exc}")
                self.set_status_threadsafe("Reindex failed", "warning")
            finally:
                self.signals.reindex_button_enabled.emit(True)

        threading.Thread(target=do_request, daemon=True).start()

    # ------------------------------------------------------------------
    # Character Memory actions
    # ------------------------------------------------------------------
    def character_memory_url(self, endpoint: str) -> str:
        server = self.settings.get("server", {})
        base = f"http://{server.get('host', 'localhost')}:{server.get('port', 5100)}"
        return f"{base}/character-memory/{endpoint.lstrip('/')}"

    def set_character_memory_buttons_enabled(self, enabled: bool) -> None:
        for button in (
            getattr(self, "char_mem_scan_btn", None),
            getattr(self, "char_mem_backup_btn", None),
            getattr(self, "char_mem_summarize_btn", None),
            getattr(self, "char_mem_update_profile_btn", None),
        ):
            if button is not None:
                button.setEnabled(enabled)

    def write_character_memory_result(self, payload: Any) -> None:
        try:
            text = json.dumps(payload, indent=2, ensure_ascii=False)
        except Exception:
            text = str(payload)
        self.char_mem_result_box.setPlainText(text)
        cursor = self.char_mem_result_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.char_mem_result_box.setTextCursor(cursor)

    def call_character_memory_endpoint(self, endpoint: str, status_text: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 3600) -> None:
        self.save_settings(and_apply=False)
        url = self.character_memory_url(endpoint)
        self.signals.character_memory_buttons_enabled.emit(False)
        self.set_status(status_text, "info")

        def worker() -> None:
            try:
                request_payload = dict(payload or {})
                request_payload["config_path"] = str(self.settings_path)
                data = json.dumps(request_payload).encode("utf-8")
                headers = {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "X-GameMaster-Config-Path": str(self.settings_path),
                }
                req = request.Request(url, data=data, headers=headers, method="POST")
                with request.urlopen(req, timeout=timeout) as response:
                    body = response.read().decode("utf-8", errors="replace")
                result = json.loads(body) if body else {}
                self.signals.character_memory_result.emit(result)
                status = str(result.get("status", "success")) if isinstance(result, dict) else "success"
                if status == "success":
                    if endpoint == "scan":
                        msg = f"Scanned {result.get('character_files', '?')} character files"
                    elif endpoint == "backup":
                        msg = f"Backed up {result.get('file_count', '?')} files"
                    elif endpoint == "summarize":
                        msg = f"Summarized {result.get('updated_files', '?')} files"
                    elif endpoint == "update-profile":
                        msg = f"Updated {result.get('updated_files', '?')} profiles"
                    else:
                        msg = "Character Memory complete"
                    self.set_status_threadsafe(msg, "success")
                elif status == "busy":
                    self.set_status_threadsafe("Character Memory busy", "warning")
                else:
                    self.set_status_threadsafe("Character Memory finished", "info")
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                print(f"Character Memory {endpoint} failed: HTTP {exc.code} {body}")
                self.signals.character_memory_result.emit({"status": "error", "http": exc.code, "body": body})
                self.set_status_threadsafe(f"Character Memory failed: HTTP {exc.code}", "warning")
            except Exception as exc:
                print(f"Character Memory {endpoint} failed: {exc}")
                self.signals.character_memory_result.emit({"status": "error", "error": str(exc)})
                self.set_status_threadsafe("Character Memory failed", "warning")
            finally:
                self.signals.character_memory_buttons_enabled.emit(True)

        threading.Thread(target=worker, daemon=True).start()

    def scan_character_memory(self) -> None:
        self.call_character_memory_endpoint("scan", "Scanning campaign…", timeout=120)

    def backup_character_memory(self) -> None:
        self.call_character_memory_endpoint("backup", "Backing up campaign…", timeout=600)

    def summarize_character_memory(self) -> None:
        payload = {"create_backup": bool(self.char_mem_backup_before_var.isChecked())}
        self.call_character_memory_endpoint("summarize", "Summarizing memories…", payload=payload, timeout=7200)

    def update_character_profiles(self) -> None:
        payload = {"create_backup": bool(self.char_mem_backup_before_var.isChecked())}
        self.call_character_memory_endpoint("update-profile", "Updating profiles…", payload=payload, timeout=7200)

    # ------------------------------------------------------------------
    # Log viewer
    # ------------------------------------------------------------------
    def on_log_viewer_toggled(self, enabled: bool) -> None:
        if enabled:
            self.log_timer.start()
            self.refresh_logs()
        else:
            self.log_timer.stop()

    def tail_file(self, raw_path: str, max_bytes: int = 280_000) -> str:
        if not raw_path.strip():
            return "No log path configured."
        path = path_from_text(raw_path)
        if not path.exists():
            return f"Log file does not exist yet:\n{path}"
        try:
            size = path.stat().st_size
            with path.open("rb") as f:
                f.seek(max(0, size - max_bytes))
                data = f.read().decode("utf-8", errors="replace")
            return data
        except Exception as exc:
            return f"Could not read {path}: {exc}"

    def on_log_scroll_changed(self, edit: QPlainTextEdit, value: int) -> None:
        """Track whether a log pane should keep following the newest entries.

        Auto-follow is enabled only while the scrollbar is at the bottom. Once
        the user scrolls up, refreshes must not move the viewport. Scrolling
        back to the bottom re-enables tail-following.
        """
        if edit.property("_gm_updating_log"):
            return
        bar = edit.verticalScrollBar()
        at_bottom = value >= max(0, bar.maximum() - 4)
        edit.setProperty("_gm_follow_tail", at_bottom)
        edit.setProperty("_gm_user_scrolled", not at_bottom)

    def update_log_text(self, edit: QPlainTextEdit, text: str) -> None:
        """Update a log pane without stealing the user's scroll position.

        The old implementation rebuilt the whole editor with setPlainText() on
        every timer tick. Qt can snap the viewport to the end while that happens,
        even if the scrollbar is restored afterward. This version appends only
        newly-added text when possible and restores the exact scrollbar value
        whenever the user is not already at the bottom.
        """
        last_text = edit.property("_gm_last_log_text")
        if not isinstance(last_text, str):
            last_text = ""
        if last_text == text:
            return

        bar = edit.verticalScrollBar()
        old_value = bar.value()
        was_at_bottom = old_value >= max(0, bar.maximum() - 4)
        should_follow_tail = bool(edit.property("_gm_follow_tail")) and was_at_bottom

        edit.setProperty("_gm_updating_log", True)
        try:
            # Fast path: log file grew normally. Mutating the document in place
            # avoids the visible jump caused by replacing all text.
            if last_text and text.startswith(last_text):
                delta = text[len(last_text):]
                if delta:
                    cursor = QTextCursor(edit.document())
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    cursor.insertText(delta)
            else:
                # First load, truncation, path change, or tail-window rollover.
                # This may rebuild the document, but it is uncommon compared with
                # normal append refreshes.
                edit.setPlainText(text)

            edit.setProperty("_gm_last_log_text", text)
            new_bar = edit.verticalScrollBar()
            if should_follow_tail:
                new_bar.setValue(new_bar.maximum())
                edit.setProperty("_gm_follow_tail", True)
                edit.setProperty("_gm_user_scrolled", False)
            else:
                new_bar.setValue(min(old_value, new_bar.maximum()))
                edit.setProperty("_gm_follow_tail", False)
                edit.setProperty("_gm_user_scrolled", True)
        finally:
            edit.setProperty("_gm_updating_log", False)

    def refresh_logs(self) -> None:
        if not hasattr(self, "log_viewer_enabled_var") or not self.log_viewer_enabled_var.isChecked():
            return
        if not hasattr(self, "llm_log_view") or not hasattr(self, "selector_log_view"):
            return
        self.update_log_text(self.llm_log_view[1], self.tail_file(self.llm_log_path_entry.text()))
        self.update_log_text(self.selector_log_view[1], self.tail_file(self.sel_log_path_entry.text()))

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def make_setting_widget(self, kind: str, value: Any, bounds: Optional[Tuple[Any, Any]]) -> QWidget:
        if kind == "bool":
            checkbox = QCheckBox()
            checkbox.setChecked(text_to_bool(value))
            return checkbox
        if kind == "float":
            return make_spin("float", value, bounds)
        if kind == "int":
            return make_spin("int", value, bounds)
        return polish_line_edit(QLineEdit(str(value or "")))

    def set_widget_value(self, widget: QWidget, value: Any, kind: str) -> None:
        if kind == "bool" and isinstance(widget, QCheckBox):
            widget.setChecked(text_to_bool(value))
        elif kind == "float" and isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value or 0.0))
        elif kind == "int" and isinstance(widget, QSpinBox):
            widget.setValue(int(value or 0))
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value or ""))

    def widget_value(self, widget: QWidget, kind: str) -> Any:
        if kind == "bool" and isinstance(widget, QCheckBox):
            return bool(widget.isChecked())
        if kind == "float" and isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        if kind == "int" and isinstance(widget, QSpinBox):
            return int(widget.value())
        if isinstance(widget, QLineEdit):
            return widget.text()
        return None

    def get_gm_dict(self) -> Dict[str, Any]:
        return deepcopy(self.settings.get("gm", {})) if isinstance(self.settings.get("gm"), dict) else {}

    def rule_list(self, rule_kind: str) -> List[Dict[str, Any]]:
        if rule_kind == "drop":
            return self.drop_rules
        if rule_kind == "replace":
            return self.replace_rules
        return self.context_rules

    def rule_schema(self, rule_kind: str) -> List[Tuple[str, str, str]]:
        if rule_kind == "drop":
            return self.drop_rule_schema()
        if rule_kind == "replace":
            return self.replace_rule_schema()
        return self.context_rule_schema()

    def strip_obsolete_rule_fields(self, rule: Dict[str, Any]) -> None:
        for key in list(rule.keys()):
            if key.startswith("delete_") and key.endswith("_line"):
                rule.pop(key, None)

    def rule_value(self, rule: Dict[str, Any], key: str) -> Any:
        value = rule.get(key)
        if key in PROMPT_RULE_TEXT_KEYS and isinstance(value, str):
            value = normalize_prompt_rule_text(value)
            rule[key] = value
        return value

    def set_rule_bool(self, rule: Dict[str, Any], key: str, value: bool) -> None:
        rule[key] = bool(value)

    def clear_layout(self, layout: QVBoxLayout | QGridLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            child = item.widget()
            child_layout = item.layout()
            if child is not None:
                child.deleteLater()
            elif child_layout is not None:
                self.clear_layout(child_layout)  # type: ignore[arg-type]

    def set_status(self, text: str, tone: str = "neutral") -> None:
        self.status_label.setText(text)
        self.status_label.setProperty("tone", tone)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def set_status_threadsafe(self, text: str, tone: str = "neutral") -> None:
        self.signals.status.emit(text, tone)

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        self.log_timer.stop()
        if self.server_process and self.server_process.poll() is None:
            self.stop_server()
        event.accept()


STYLE = """
QWidget {
    background: #0f1117;
    color: #d7deea;
    font-family: "Segoe UI", "Inter", Arial, sans-serif;
    font-size: 12px;
}
QFrame#HeaderBar {
    background: #151924;
    border: 1px solid #242b3a;
    border-radius: 14px;
}
QFrame#Card {
    background: #151924;
    border: 1px solid #242b3a;
    border-radius: 14px;
}
QLabel#Title {
    color: #edf2fb;
    font-size: 21px;
    font-weight: 760;
}
QLabel#Subtitle, QLabel#Hint {
    color: #8b96aa;
}
QLabel#TinyHint {
    color: #707b8f;
    font-size: 11px;
}
QLabel#CardTitle {
    color: #edf2fb;
    font-size: 14px;
    font-weight: 700;
}
QLabel#FieldLabel {
    color: #cfd8e8;
    font-weight: 600;
}
QLabel#StatusPill {
    background: #171c29;
    border: 1px solid #2a3344;
    border-radius: 11px;
    padding: 5px 10px;
    color: #aab4c6;
    min-width: 76px;
}
QLabel#StatusPill[tone="success"] { color: #a7f3d0; border-color: #285743; background: #101d19; }
QLabel#StatusPill[tone="warning"] { color: #f6d98f; border-color: #6f5222; background: #211a10; }
QLabel#StatusPill[tone="danger"] { color: #f5b5b5; border-color: #713131; background: #221414; }
QLabel#StatusPill[tone="info"] { color: #b7d8ff; border-color: #34537d; background: #111b2a; }
QTabWidget::pane {
    border: 1px solid #242b3a;
    border-radius: 13px;
    padding: 6px;
    background: #101620;
}
QTabBar::tab {
    background: #151b28;
    border: 1px solid #242c3d;
    color: #9ca8bb;
    padding: 7px 12px;
    min-height: 18px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 3px;
}
QTabBar::tab:selected {
    background: #20293b;
    color: #eef3fb;
    border-color: #3f516f;
}
QTabBar::tab:hover { background: #1b2333; }
QTabWidget#InnerTabs::pane { border-radius: 12px; }
QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #0b0e14;
    color: #edf2fb;
    border: 1px solid #273044;
    border-radius: 8px;
    padding: 4px 8px;
    selection-background-color: #435d8d;
    min-height: 24px;
}
QComboBox::drop-down {
    border: none;
    width: 26px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #8b96aa;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background: #111722;
    color: #edf2fb;
    border: 1px solid #2a3344;
    selection-background-color: #273752;
    selection-color: #ffffff;
}
QPlainTextEdit {
    padding: 8px;
    line-height: 1.25em;
}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #5d78a6;
    background: #0d121c;
}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 0px;
    border: 0px;
}
QPushButton {
    background: #1a2230;
    border: 1px solid #30394d;
    border-radius: 9px;
    padding: 5px 10px;
    color: #dce5f4;
    font-weight: 600;
}
QPushButton:hover { background: #202b3d; border-color: #44546f; }
QPushButton:pressed { background: #141b28; }
QPushButton:disabled { color: #647084; background: #141923; border-color: #1d2532; }
QPushButton#PrimaryButton {
    background: #2d405f;
    border-color: #58739d;
    color: #edf2fb;
}
QPushButton#PrimaryButton:hover { background: #354c70; }
QPushButton#DangerButton {
    background: #4b2528;
    border-color: #794044;
    color: #f5d5d8;
}
QPushButton#DangerButton:hover { background: #5a2d31; }
QCheckBox {
    spacing: 8px;
    color: #cfd8e8;
    min-height: 26px;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border-radius: 5px;
    border: 1px solid #445066;
    background: #0b0e14;
}
QCheckBox::indicator:checked {
    background: #516b98;
    border: 1px solid #7e98c3;
}
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical, QScrollBar:horizontal {
    background: #0f141d;
    border: none;
    width: 10px;
    height: 10px;
    margin: 0px;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #2b3549;
    border-radius: 5px;
    min-height: 26px;
    min-width: 26px;
}
QScrollBar::handle:hover { background: #3a465c; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0px; height: 0px; }
QSplitter::handle { background: #242b3a; width: 5px; }
"""

def run_embedded_server() -> None:
    """Run the FastAPI backend from the same frozen EXE.

    The GUI launches this by calling: GMPython.exe --server
    This keeps distribution to a single EXE while still allowing the GUI to
    stop/restart the backend process.
    """
    os.chdir(APP_ROOT)
    if not SETTINGS_PATH.exists():
        write_default_settings_if_missing(SETTINGS_PATH)

    from config.settings import Settings
    import main as server_main
    import uvicorn

    settings = Settings.load(str(SETTINGS_PATH))
    ensure_console_streams()
    uvicorn.run(
        server_main.app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        access_log=False,
        log_config=None,
        use_colors=False,
    )


def main() -> None:
    if "--server" in sys.argv:
        run_embedded_server()
        return
    app = QApplication(sys.argv)
    app.setApplicationName("GameMaster")
    app.setStyleSheet(STYLE)
    win = GameMasterGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
