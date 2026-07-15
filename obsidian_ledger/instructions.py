"""Write the consult-the-ledger instruction into every agent's context
file, idempotently (marked block). One `install` covers Claude Code,
Codex, Gemini CLI, Cursor, Windsurf, OpenCode, Antigravity, Grok Code
and anything else that reads AGENTS.md or executes shell.
"""
from __future__ import annotations

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

START = "<!-- obsidian-ledger:start -->"
END = "<!-- obsidian-ledger:end -->"

BLOCK = f"""{START}
## Vault Ledger (source of truth for facts)

This vault has a symbolic fact ledger at `_ledger/`:

- `_ledger/VAULT_LEDGER.md` — CURRENT facts, one per line, each with a
  date and a source note. Superseded values are listed separately.
- `_ledger/vault_ledger.jsonl` — full structured history.

**Rules for any agent working in this vault:**
1. Before asserting any fact from this vault (an ID, a DOI, an active
   rule, a metric, a status), read `_ledger/VAULT_LEDGER.md` first — it
   is one short page.
2. If the fact is there, use the CURRENT value (never a superseded one).
3. If the fact is NOT there, do not assert it — verify at the source
   note or say you don't know. Notes in this vault deliberately keep
   outdated versions; the ledger is where supersession is resolved.
4. CLI (if you can run shell), from the ledger repo at
   `{REPO}`:
   `PYTHONPATH="{REPO}" python3 -m obsidian_ledger.cli query
   "<question>" --vault <vault-path>` (also: `verify "<claim>"`,
   `induce --entity E --attribute A`). Answers carry provenance or an
   explicit NOT_IN_LEDGER.
{END}"""

# context file per harness; AGENTS.md is the emerging cross-agent standard
TARGETS = ["AGENTS.md", "CLAUDE.md", "GEMINI.md", ".cursorrules",
           ".windsurfrules"]


def install(vault: str) -> list:
    written = []
    for name in TARGETS:
        path = os.path.join(vault, name)
        existing = open(path).read() if os.path.exists(path) else ""
        if START in existing:
            pre = existing.split(START)[0]
            post = existing.split(END)[-1]
            content = pre + BLOCK + post
        else:
            sep = "\n\n" if existing and not existing.endswith("\n\n") \
                else ""
            content = existing + sep + BLOCK + "\n"
        with open(path, "w") as f:
            f.write(content)
        written.append(name)
    return written
