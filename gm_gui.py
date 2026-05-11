#!/usr/bin/env python3

# Copyright (C) 2026 SEGAREGA
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compatibility launcher for the GameMaster GUI.

The real GUI lives in gamemaster_gui.py. This file is intentionally tiny so old
shortcuts or commands that launch gm_gui.py keep working without maintaining a
second copy of the UI.
"""

from gamemaster_gui import main

if __name__ == "__main__":
    main()
