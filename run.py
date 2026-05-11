# Copyright (C) 2026 SEGAREGA
# SPDX-License-Identifier: AGPL-3.0-or-later

"""GameMaster - Server Launcher."""

from __future__ import annotations

import os
import sys

# Add current directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import Settings


def main() -> None:
    print("=" * 50)
    print("GameMaster - Starting Server")
    print("=" * 50)
    print()

    settings = Settings.load()
    print(f"Configuration: {settings.config_path or 'defaults'}")
    print(f"Backend: {settings.get_chat_completions_url()}")
    print(f"Model: {settings.models}")
    print(f"Server: http://{settings.host}:{settings.port}")
    print()

    if settings.uses_selector_filtering() and (
        not settings.selector_enabled or not settings.selector_api_url or not settings.selector_model
    ):
        print("ERROR: selector settings are incomplete.")
        print("Please configure the 'selector' block in config/settings.json.")
        print('- "enabled": true')
        print('- "api_url": "http://localhost:1234/v1" (or your OpenAI-compatible endpoint)')
        print('- "model": "Qwen/Qwen2.5-7B-Instruct" (or your selector model)')
        print()
        sys.exit(1)

    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
