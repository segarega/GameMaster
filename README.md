# GameMaster for AI Influence

**GameMaster** is a local OpenAI-compatible proxy and GUI for the **AI Influence** Mount & Blade II: Bannerlord mod. It filters, indexes, and injects only the relevant lore, rules, memory, and game context before each LLM request.

This helps AI Influence use fewer input tokens, stay more coherent, and support larger custom worlds without sending your entire `world.txt` and rule files on every request.

---

## Contents

- [Why Use GameMaster?](#why-use-gamemaster)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Required `.txt` File Format](#required-txt-file-format)
- [GUI Reference](#gui-reference)
  - [General](#general)
  - [GM Filtering](#gm-filtering)
  - [Request Parameters](#request-parameters)
  - [System Prompts](#system-prompts)
  - [Selector](#selector)
  - [Detection](#detection)
  - [Drop Rules](#drop-rules)
  - [Replace Rules](#replace-rules)
  - [Context Rules](#context-rules)
  - [Character Memory](#character-memory)
  - [Log Viewer](#log-viewer)
- [Backend Architecture](#backend-architecture)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Credits](#credits)

---

## Why Use GameMaster?

AI Influence can send a lot of information to the LLM: world lore, character data, rules, histories, settlements, parties, and prior conversations. Sending everything every time can waste tokens and make the model less focused.

GameMaster sits between AI Influence and your LLM provider. It trims the prompt, selects relevant lore, compresses character memory, and forwards a cleaner request to your chosen model.

### Main benefits

| Benefit | What it means |
|---|---|
| **Lower token usage** | Send only relevant `[GM]` sections instead of full lore/rule files. |
| **Better coherence** | The LLM receives a smaller, cleaner context instead of unrelated information. |
| **Cheaper model routing** | Use a cheap/fast Selector model to decide what your stronger main model should see. |
| **Persistent character memory** | Summarize old conversations while preserving recent dialogue. |
| **More control** | Drop, replace, or extract prompt sections using GUI rules. |

---

## Key Features

- 🎮 **Built for AI Influence** — understands the prompt structure used by the Bannerlord mod.
- 🤖 **Dual-stage prompting** — a Selector LLM chooses relevant `[GM]` content before the main LLM request.
- 📚 **Static GM Index** — indexes `[GM]` entries from files such as `world.txt` and `actionrules.txt`.
- 🧠 **Character Memory compression** — summarizes old conversation history into compact `MEMORY` entries.
- 📝 **Prompt rules engine** — drop, replace, and extract prompt sections using marker-based rules.
- 🔌 **OpenAI-compatible** — works with OpenAI, OpenRouter, NanoGPT, LM Studio, KoboldCPP, text-generation-webui, and similar `/chat/completions` providers.
- 🖥️ **PyQt6 GUI** — dark-themed settings editor with live logs and model fetching.
- 🔥 **Partial hot reload** — some settings can be changed without restarting the game or proxy.

---

## How It Works

```text
AI Influence  →  GameMaster Proxy  →  LLM Provider
                    │
                    ├─ Filters prompt sections
                    ├─ Selects relevant [GM] lore/rules
                    ├─ Injects [PINNED] content
                    ├─ Drops [IGNORE] content
                    └─ Compresses character memory
```

GameMaster can use two different LLM calls:

| Stage | Purpose | Model note |
|---|---|---|
| **Selector LLM** | Chooses which indexed `[GM]` sections are relevant. | A cheap, fast, deterministic model is usually enough. |
| **Main LLM** | Generates the final dialogue, diplomacy, or event response. | Use whichever main model you configured for that request type. |

### Example savings setups

| Setup | Example use |
|---|---|
| **NanoGPT subscription** | Use app filtering plus offloaded GM selection to reduce total tokens sent to your main model. |
| **OpenRouter** | Use a cheaper model for selection and a stronger model for final output. |
| **Hybrid NanoGPT/OpenRouter** | Use a small paid OpenRouter model only for selection/summaries while spending NanoGPT allowance on the main model. |
| **Hybrid local/cloud** | Use a local GPU model for selection and a cloud model for final output. |

---

## Installation

### Requirements

- Python **3.10+ (IF not using .exe from release)**
- Python added to your system `PATH` (IF not using .exe from release)
- AI Influence installed for Mount & Blade II: Bannerlord

### Option 1: Download the `.exe` release

Recommended for most users.

1. Download the latest release.
2. Run `GameMaster.exe`.
3. On first launch, GameMaster creates its `config` and `cache` folders.
4. On first **Save & Reindex DB**, it creates the local database cache.

### Option 2: Build the GUI `.exe`

From the project root, run:

```bat
build_onefile_gui.bat
```

This creates:

```text
dist/GameMaster.exe
```

Move `GameMaster.exe` wherever you want. It creates the required `config` and `cache` folders on first run.

### Option 3: Run without the GUI

Run:

```bat
start.bat
```

> [!NOTE]
> Headless mode requires manually editing `settings.json`.

---

## Quick Start

1. **Launch GameMaster.**
2. Open the **General** tab.
3. Fill in your main LLM settings:
   - **API URL**
   - **API Key**
   - **Model IDs** or click **Fetch Models**
4. Open the **Selector** tab.
5. Fill in the Selector settings:
   - **API URL**
   - **API Key**
   - **Model**
6. Set your **AIInfluence folder** to:

   ```text
   .../Mount & Blade II Bannerlord/Modules/AIInfluence/
   ```

7. Click **Save & Reindex DB**.
8. Click **Save**, then **Stop**, then **Start**.
9. In AI Influence mod settings, choose **Player2** and set the port to the GameMaster port, usually:

   ```text
   localhost:5100
   ```

> [!IMPORTANT]
> You must click **Save & Reindex DB** after editing your AI Influence `.txt` files. Otherwise, the database may be empty or outdated.

---

## Required `.txt` File Format

GameMaster reads policy headers inside AI Influence text files. These headers tell GameMaster which content should always be kept, selected dynamically, or ignored.

### Policy headers

| Header | Behavior |
|---|---|
| `[GM]` | Selector-controlled. The content is indexed and only included when relevant. |
| `[PINNED]` | Always included. Use for critical rules, world setup, or mission statements. |
| `[IGNORE]` | Always removed. Use for drafts, comments, or disabled content. |
| `[END GM]`, `[END PINNED]`, `[END IGNORE]` | Closes the final policy section at the **end of the entire `.txt` file**. You usually do **not** need these inside the file, because each new policy header automatically ends the previous one. |

> [!IMPORTANT]
> Use `[END GM]`, `[END PINNED]`, or `[END IGNORE]` at the **end of the whole `.txt` file**, after the final policy section.
>
> Do **not** use `[END ...]` markers as normal separators inside the file. Inside the file, `[GM]`, `[PINNED]`, and `[IGNORE]` headers already end each other.
>
> The end marker matters because AI Influence appends its own prompt immediately after your text file content. If your file ends while still inside a `[PINNED]`, `[GM]`, or `[IGNORE]` section, the mod's appended prompt may accidentally become part of that final section.

Example final entry:

```txt
[PINNED] Technical Rules
These rules must always be sent.

[END PINNED]
```

### Scoped headers

You can limit headers to specific request types.

| Header | Applies to |
|---|---|
| `[GM:DIALOGUE]` | Dialogue requests only. |
| `[GM:EVENTS]` | Event-generation requests only. |
| `[GM:DIPLOMACY]` | Diplomacy requests only. |
| `[PINNED:DIPLOMACY:EVENTS]` | Always included for diplomacy and events. |

### GM child elements

Inside `[GM]` sections, `##` is reserved for child elements under a parent block title.

> [!IMPORTANT]
> `##` is a child element of `[GM]` sections and **must** be used for child elements of a specific GM block.
>
> `##` **must not** be used for anything else in your `.txt` files.

You may still use `#`, `####`, `=====`, or similar custom separators outside GM child elements. They are skipped by the GM query logic.

### Example `world.txt`

```txt
[PINNED] *** World Lore ***
You are roleplaying in Game of Thrones 283 AC universe. xyz

[GM:DIALOGUE] == Cultures ==
## Braavosi are xyz
They are also xyz

## Tyroshi are xyz
They are also...

## Westerlands are xyz
They are also...

```

In this example:

| Item | Meaning |
|---|---|
| `*** World Lore ***` | Pinned content that is always sent because its header is `[PINNED]` with no request type after `:`. |
| `== Cultures ==` | GM block title for dialogue requests only, because its header is `[GM:DIALOGUE]`. |
| `## Braavosi are xyz` | GM child element under the `== Cultures ==` block. |
| `## Tyroshi are xyz` | GM child element under the `== Cultures ==` block. |
| `## Westerlands are xyz` | GM child element under the `== Cultures ==` block. |

If the current conversation only concerns the Westerlands, the request to your main LLM will include only this:

```txt
*** World Lore ***
You are roleplaying in Game of Thrones 283 AC universe. xyz

== Cultures ==
## Westerlands are xyz
They are also...
```

The pinned lore is always included. Irrelevant Braavosi and Tyroshi entries are skipped.

> [!TIP]
> The included example `world.txt` is meant as a reference. Most users only need to edit headers and titles in their existing AI Influence files.

---

## GUI Reference

### General

The **General** tab controls server settings, the main LLM backend, model routing, and logging.

#### Server

| Setting | Description |
|---|---|
| **Host** | The interface the proxy binds to. `localhost` is safest. Use `0.0.0.0` only if you need remote access. |
| **Port** | TCP port used by the proxy. Default: `5100`. |

> [!IMPORTANT]
> In AI Influence, choose **Player2** and set its port to the same port configured here.

#### LLM Backend

| Setting | Description |
|---|---|
| **API URL** | Main LLM API URL. Examples: `https://api.openai.com/v1`, `https://openrouter.ai/api/v1`, `https://nano-gpt.com/api/subscription/v1/chat/completions`, `http://localhost:1234/v1`. |
| **API Key** | Authentication token. Stored locally in `config/settings.json`. |
| **Site URL / Referer** | Optional `HTTP-Referer` header. Useful for some providers, such as OpenRouter. |
| **App Title** | Optional `X-Title` header identifying your app to the provider. |

#### Models by request type

AI Influence sends different kinds of requests. GameMaster lets you assign a separate main LLM model for each one.

| Request type | Used for |
|---|---|
| **Events** | Narrative event generation. |
| **Diplomacy** | Kingdom statements, faction logic, and political declarations. |
| **Dialogue** | In-character NPC conversation. |

Click **Fetch Models** to query the provider's `/models` endpoint and populate the dropdowns.

#### Logging

| Setting | Description |
|---|---|
| **Live log viewer** | Enables automatic log refreshing in the Log Viewer tab. |
| **LLM logging** | Logs requests and responses sent to the main LLM backend. |
| **LLM log path** | File path for the main LLM log. |
| **LLM pretty JSON** | Formats logged JSON with indentation. |
| **Selector logging** | Logs Selector requests and responses. |
| **Selector log path** | File path for Selector logs. |
| **Selector pretty JSON** | Formats Selector logs with indentation. |

---

### GM Filtering

The **GM Filtering** tab controls dynamic prompt pruning before the Selector and main LLM see the prompt.

> [!NOTE]
> Prompt policy headers such as `[GM]`, `[PINNED]`, and `[IGNORE]` are still respected. This tab mainly controls numeric limits and algorithmic filters.

#### Core GM filtering

| Setting | Description |
|---|---|
| **Enable dynamic GM filtering** | Master switch. When disabled, prompts pass through mostly unmodified, apart from policy-header logic. |
| **Disable User's last message during NPC-NPC conversation** | When enabled, dialogue requests containing `===== NPC-TO-NPC CONVERSATION MODE =====` have outbound `role: user` messages removed before sending to the main LLM. This prevents the mod's starter instruction from making NPC-to-NPC exchanges restart repeatedly. Default: `disabled`. |
| **Disable User's last message during Group Chat** | When enabled, dialogue requests containing `===== GROUP CONVERSATION MODE =====` have outbound `role: user` messages removed before sending to the main LLM. Use this when group conversations include a leftover user starter instruction that should not guide the NPC exchange. Default: `disabled`. |
| **Fuzzy match threshold** | `0.0`–`1.0`. Higher values make dynamic relevance matching stricter for `Hide-Until-Relevant Controls`. Default: `0.88`. |
| **Max event history** | Maximum historical events kept in the prompt. Default: `200`. |
| **Events — max dialogue lines** | Caps embedded dialogue excerpts in event requests. Default: `20`. |
| **Events — max settlements mentioned** | Caps settlement references in event prompts. Default: `10`. |

Dialogue history length is controlled by AI Influence's own dialogue history setting. GameMaster does not apply a second dialogue-history cap.

#### Dynamic context limits

| Setting | Description |
|---|---|
| **Max people present** | Maximum NPCs listed in the current location. Default: `10`. |
| **Max nearby settlements** | Maximum settlement descriptions. Default: `7`. |
| **Max nearby parties** | Maximum wandering party/warband descriptions. Default: `5`. |
| **Max inventory lines** | Maximum inventory lines. Default: `5`. |

#### Hide-Until-Relevant Controls

| Setting | Description |
|---|---|
| **Character Briefing (CURRENT DATA)** | Show/Hide this section from request until relevant based on word similarity. Default: `disabled`. |
| **The Player Current Data** | Show/Hide this section from request until relevant based on word similarity. Default: `disabled`. |
| **People physically present** | Show/Hide this section from request until relevant based on word similarity. Default: `disabled`. |
| **Nearby settlements** | Show/Hide this section from request until relevant based on word similarity. Default: `disabled`. |
| **Nearby parties** | Show/Hide this section from request until relevant based on word similarity. Default: `disabled`. |
| **Mentioned settlements** | Show/Hide this section from request until relevant based on word similarity. Default: `disabled`. |
| **Mentioned characters** | Show/Hide this section from request until relevant based on word similarity. Default: `disabled`. |
| **Mentioned parties** | Show/Hide this section from request until relevant based on word similarity. Default: `disabled`. |
| **Appearance/equipment lines** | Show/Hide this section from request until relevant based on word similarity. Default: `disabled`. |
| **Wealth/money lines** | Show/Hide this section from request until relevant based on word similarity. Default: `disabled`. |
| **Inventory/item lines** | Show/Hide this section from request until relevant based on word similarity. Default: `disabled`. |
| **Clan line** | Show/Hide this section from request until relevant based on word similarity. Default: `disabled`. |
| **Family/relatives lines** | Show/Hide this section from request until relevant based on word similarity. Default: `disabled`. |
| **Relations/friends/enemies lines** | Show/Hide this section from request until relevant based on word similarity. Default: `disabled`. |
| **Forces/troops lines** | Show/Hide this section from request until relevant based on word similarity. Default: `disabled`. |
| **Captives/prisoners lines** | Show/Hide this section from request until relevant based on word similarity. Default: `disabled`. |
| **Workshop/business lines** | Show/Hide this section from request until relevant based on word similarity. Default: `disabled`. |

---

### Request Parameters

The **Request Parameters** tab controls sampling settings per request type.

> [!IMPORTANT]
> Unchecked parameters are omitted from the outbound API request. This lets your provider use its own defaults.

| Parameter | Type | Description |
|---|---|---|
| **temperature** | Float | `0`–`2`. Lower is more deterministic; higher is more creative/random. |
| **top_p** | Float | `0`–`1`. Nucleus sampling cutoff. `1.0` effectively disables it. |
| **top_k** | Integer | Provider-specific. Common in local models. Strict OpenAI-style APIs may ignore it. Default: `40`. |

Only checked parameters are sent. Leave a parameter unchecked if you want your provider or local backend to use its own default behavior.

---

### System Prompts

The **System Prompts** tab lets you inject extra system messages before or after the intercepted prompt history.

| Field | Inserted where? | Best used for |
|---|---|---|
| **Pre-history system prompt** | First system message. | Global style, tone, or formatting rules. |
| **Post-history system prompt** | Last system message. | Final reminders, constraints, or response rules. |

> [!WARNING]
> Do not put `[GM]`, `[PINNED]`, or `[IGNORE]` sections here. Those policy headers are for AI Influence `.txt` files, not GUI system prompts.

Empty fields are ignored.

---

### Selector

The **Selector** tab controls the relevance engine, the Static GM Index, and the reindexing pipeline.

#### Selector model

| Setting | Description |
|---|---|
| **API URL** | Selector API URL. If blank, defaults to the main LLM API URL. A cheaper/faster endpoint is recommended. |
| **API Key** | Selector API key. If blank, defaults to the main API key. |
| **Model** | Model used to decide which `[GM]` sections are relevant. Cheap and fast models are usually enough. |
| **Temperature** | Selector temperature. Default: `0.0` for deterministic choices. |
| **Max tokens** | Maximum Selector JSON response size. Default: `32000`. Increase if you have many GM sections. |
| **Timeout seconds** | Maximum wait time for the Selector. Default: `120`. |

#### Selector instruction

This is the system prompt sent to the Selector. It defines how candidate entries should be evaluated and what JSON shape the Selector must return.

> [!CAUTION]
> Only edit this if you know what you are changing. A small instruction change can significantly affect selection quality.

If you write a custom Selector instruction, it must tell the LLM to output JSON in this format:

```json
{"blocks":[{"block_id":"block_1","keep_ids":["id1","id2"]}]}
```

#### Static GM Index

The Static GM Index reads AI Influence `.txt` files and stores summaries of `[GM]` sections in a local SQLite database.

| Setting | Description |
|---|---|
| **Static GM index** | Master toggle. Must be enabled for `[GM]` headers to work. |
| **AIInfluence folder** | Path to `.../Mount & Blade II Bannerlord/Modules/AIInfluence/`. |
| **Indexed files** | Comma-separated files to index, such as `world.txt,actionrules.txt`. |
| **DB path** | SQLite cache path. Default: `cache/static_gm_index.sqlite3`. |
| **LLM summaries** | Uses an LLM to generate compact summaries for each `[GM]` child section. Recommended. |
| **Summary model override** | Optional model specifically for summaries. Blank means use the Selector model. |
| **Summary prompt** | Instruction used during reindexing to create dense summaries. |

> [!IMPORTANT]
> Click **Save & Reindex DB** after the first setup and every time you edit indexed `.txt` files.

---

### Detection

The **Detection** tab defines strings used to identify the request type: dialogue, diplomacy, or events.

GameMaster uses detection to choose:

1. The correct model.
2. The correct request-parameter overrides.
3. The correct custom system prompts.
4. The correct scoped `[PINNED]`, `[IGNORE]`, and `[GM]` sections.

| Field | Description |
|---|---|
| **Request type** | `dialogue`, `events`, or `diplomacy`. |
| **Detection string** | A substring that uniquely appears in that request type. Can be multi-line. |

Default dialogue detection includes the standard mission prompt, `===== GROUP CONVERSATION MODE =====`, and `===== NPC-TO-NPC CONVERSATION MODE =====`, so group conversations and NPC-to-NPC conversations resolve as dialogue.

> [!WARNING]
> Do not change the defaults unless you have **dropped** or **replaced** the matching AI Influence prompt markers with GameMaster rules.

---

### Drop Rules

Drop Rules delete prompt sections between two marker strings.

| Field | Description |
|---|---|
| **Rule name** | Human-readable name. |
| **Request types** | Comma-separated types, such as `dialogue,events`, or `all`. |
| **Drop beginning marker** | Text that marks the start of the section to delete. |
| **Drop end marker** | Text that marks the end of the section to delete. |
| **Delete beginning marker** | Also remove the beginning marker line. |
| **Delete end marker** | Also remove the ending marker line. |

> [!NOTE]
> Drop Rules are fail-closed. If either marker is missing, nothing is deleted.

---

### Replace Rules

Replace Rules work like Drop Rules, but replace the matched section with custom text.

| Field | Description |
|---|---|
| **Rule name** | Human-readable name. |
| **Request types** | Request types this rule applies to. |
| **Replace beginning marker** | Start marker of the section to replace. |
| **Replace end marker** | End marker of the section to replace. |
| **Replacement text** | Text inserted in place of the matched section. |
| **Delete beginning marker** | Remove the beginning marker. |
| **Delete end marker** | Remove the ending marker. |

Use Replace Rules to swap default AI Influence instructions with your own custom prompts, or to compress repetitive sections into shorter text.

---

### Context Rules

Context Rules tell GameMaster what parts of the intercepted prompt should be sent to the Selector as request context.

The Selector needs high-signal information, such as:

- Who is speaking?
- Where are they?
- What is happening right now?
- What was recently said?

| Field | Description |
|---|---|
| **Rule name** | Human-readable label sent as a Selector context header. |
| **Request types** | Request types this extraction applies to. |
| **Context beginning marker** | Start of the section to extract. |
| **Context end marker** | End of the section to extract. |
| **Include beginning marker** | Keep the start marker in the extracted text. |
| **Include end marker** | Keep the end marker in the extracted text. |

Default Context Rules are provided for **Character Briefing** and **Conversation History**.

---

### Character Memory

The **Character Memory** tab compresses AI Influence `ConversationHistory` entries stored in campaign character JSON files.

Over a long campaign, conversation history can become very large. GameMaster can summarize older lines while keeping recent dialogue intact.

#### Campaign data

| Setting | Description |
|---|---|
| **Character Memory** | Master toggle. |
| **Campaign folder** | Path to `.../Modules/AIInfluence/save_data/<campaign_id>/`. |

#### Memory LLM

| Setting | Description |
|---|---|
| **API URL** | Defaults to Selector, then Main LLM, if blank. |
| **API Key** | Defaults up the chain if blank. |
| **Model** | Defaults up the chain if blank. |
| **Temperature** | Default: `0.1` for factual summaries. |
| **Max tokens** | Maximum summary length. Default: `32000`. |
| **Timeout seconds** | Maximum wait time. Default: `180`. |

#### Conversation summarization

| Setting | Description |
|---|---|
| **Preserve last raw lines** | Recent conversation lines to keep verbatim. Default: `20`. |
| **Auto mode** | Automatically scans and summarizes when conditions are met. |
| **Auto trigger raw lines** | Triggers auto-summary when a character exceeds this many raw lines. Default: `30`. |
| **Auto scan interval** | How often to check campaign files. Default: `30` seconds. |
| **Auto debounce** | Minimum time between auto-summary runs. Default: `8` seconds. |
| **Merge MEMORY entries after** | Maximum `MEMORY1`, `MEMORY2`, etc. entries to keep before merging them into one consolidated memory. Default: `5`. |

#### Prompts

| Prompt | Purpose |
|---|---|
| **Conversation summary prompt** | Compresses old dialogue into a `MEMORY` paragraph. |
| **Memory merge prompt** | Consolidates multiple `MEMORY` entries into one entry when the merge threshold is exceeded. |
| **Personality / backstory update prompt** | Decides whether character `personality` or `backstory` fields should be updated based on evidence. |

#### Actions

| Button | What it does |
|---|---|
| **Scan Campaign** | Reads character files and reports conversation sizes. |
| **Backup Current Campaign Data** | Copies character JSONs to a timestamped backup. |
| **Summarize Conversation History** | Compresses older conversation lines into `MEMORY` entries. |
| **Update Character Personality / Backstory** | Conservatively updates durable character profile details. |

> [!CAUTION]
> Always back up your campaign data before running destructive or large-scale memory operations.

---

### Log Viewer

The **Log Viewer** tab shows live logs when enabled in the General tab.

| Pane | Shows |
|---|---|
| **LLM Request/Response Log** | Main LLM requests and responses formatted according to the Pretty JSON setting. |
| **Selector Log** | Selector prompts, candidate lists, selected IDs, and provider JSON formatted according to the Pretty JSON setting. |

Auto-scroll behavior:

- If you are at the bottom, new log entries stay in view.
- If you scroll up, auto-scroll pauses until you return to the bottom.

---

## Backend Architecture

```text
┌─────────────────┐     intercepts      ┌─────────────────────┐
│  AI Influence   │ ──────────────────► │  GameMaster Proxy   │
│   Bannerlord    │  OpenAI-compatible  │  FastAPI + uvicorn  │
└─────────────────┘                     └─────────────────────┘
                                                │
                     ┌──────────────────────────┼──────────────────────────┐
                     ▼                          ▼                          ▼
            ┌─────────────┐            ┌──────────────┐           ┌─────────────┐
            │   Prompt    │            │   Selector   │           │  Character  │
            │   Filter    │            │   Client     │           │   Memory    │
            │ rules + GM  │            │ chooses GM   │           │   Manager   │
            │ relevance   │            │ blocks       │           │ compresses  │
            └─────────────┘            └──────────────┘           │  history    │
                                                                  └─────────────┘
                                                │
                                                ▼
                                       ┌────────────────┐
                                       │  Upstream LLM  │
                                       │ OpenAI / etc.  │
                                       └────────────────┘
```

### Important files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app. Handles `/v1/chat/completions`, `/reload`, `/reindex`, `/health`, and `/character-memory/*`. |
| `gamemaster_gui.py` | PyQt6 settings editor and server manager. |
| `config/settings.py` | Runtime `Settings` dataclass with load/save/normalization logic. |
| `gm/prompt_filter.py` | Prompt parsing, policy headers, dynamic filtering, and rule application. |
| `gm/selector.py` | HTTP client for the Selector LLM and JSON parsing. |
| `gm/retriever.py` | Indexes AI Influence text files into memory/SQLite and generates summaries. |
| `character_memory/manager.py` | Scans, backs up, summarizes, and updates campaign character JSONs. |

---

## Troubleshooting

### Fetch Models fails

- Verify your **API URL** ends with the provider's expected path, such as `/v1` or `/v1/chat/completions`.
- Verify your **API Key** is correct and active.
- If using LM Studio or another local server, confirm it is running and the port is correct.
- Check the **Log Viewer** for HTTP errors.

### Selector returns empty or malformed JSON

- Use a model that follows JSON instructions well.
- Set **Selector Temperature** to `0.0`.
- Check the **Selector Log** to inspect the exact prompt and response.
- Make sure the **Selector Instruction** was not accidentally truncated.
- Increase **Max tokens** if you have many GM sections.

### `[GM]` content is missing from prompts

- Confirm **Static GM index** is enabled.
- Confirm **AIInfluence folder** points to the correct directory.
- Click **Save & Reindex DB** after editing `.txt` files.
- Confirm your files contain valid `[GM]` headers.
- Confirm your scoped headers match the detected request type.

### Character Memory does nothing

- Confirm **Character Memory** is enabled.
- Confirm **Campaign folder** points to:

  ```text
  .../Modules/AIInfluence/save_data/<campaign_id>/
  ```

- Confirm the folder contains character `*.json` files.
- Run **Scan Campaign** first to verify files are detected.

### Some settings require restart

GameMaster supports partial hot reload, but not every setting can be safely applied while the proxy is already running.

- For the most reliable result after major settings changes, click **Save**, then **Stop**, then **Start**.
- If a setting appears unchanged after reload, restart the managed server from the GUI.
- If running headless, stop and restart the backend process after major config edits.
- Environment variables starting with `GMR_SELECTOR_` or `GMR_MODEL_` can override saved settings outside the GUI-managed server.

### Not sure what goes where?

Check the `Examples` folder. It includes example `settings.json` and `world.txt` files that show how the structure should look.

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 or later**.

You may use, modify, fork, and redistribute this project, but modified versions must also be licensed under AGPLv3-or-later.

If you run a modified version as a network service, you must provide users access to the corresponding source code.

You must preserve the author attribution notice for **SEGAREGA** as described in `ATTRIBUTION.md`.

---

## Credits

- **MFiveM5** — creator of the AI Influence mod.
- **Bhelogan** — idea for localdb storing of GM ids instead of from request interception.
- Great modders & users from the AI Influence community whose ideas inspired this project.

---

## Support the Project

If GameMaster helps you, consider:

- Starring the repository.
- Sharing your prompt rules and character-memory tuning tips.
- Reporting bugs with relevant log excerpts.
