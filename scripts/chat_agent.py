#!/usr/bin/env python3
"""
Terminal chat with the biomedical agent.

Uses the LangGraph Server backend and streams responses in the terminal.

Usage:
  python scripts/chat_agent.py --server-url http://localhost:2024 --user-id alice
  # or: biomedagent-db chat [options]

Commands in the REPL:
  /new                 Start a new conversation (new thread).
  /reports             List stored reports for the user.
  /load_report <id>    Load and display one report.
  /add_to_context ...  Add extra context for next messages.
  /quit, /exit   Exit.
"""

from bioagent.cli.chat import main

if __name__ == "__main__":
    main()
