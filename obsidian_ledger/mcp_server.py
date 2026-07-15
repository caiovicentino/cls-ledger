"""Minimal MCP (Model Context Protocol) stdio server for the vault
ledger — stdlib only, newline-delimited JSON-RPC 2.0.

Works with every MCP-capable harness (Claude Code/Desktop, Cursor,
Gemini CLI, Codex, OpenCode, Windsurf, ...). Register e.g.:

  claude mcp add vault-ledger -- python3 -m obsidian_ledger.mcp_server \\
      --vault "/path/to/vault"

Tools: ledger_query, ledger_verify, ledger_induce, ledger_current.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import store

TOOLS = [
    {"name": "ledger_query",
     "description": "Look up current facts in the vault ledger by "
                    "free-text question. Returns facts with value, date "
                    "and source note, or NOT_IN_LEDGER.",
     "inputSchema": {"type": "object", "properties": {
         "question": {"type": "string"}}, "required": ["question"]}},
    {"name": "ledger_verify",
     "description": "Verify a claim against the ledger: SUPPORTED, "
                    "CONTRADICTED_STALE (matches a superseded value), or "
                    "NOT_IN_LEDGER (do not assert). Anti-fabrication "
                    "check before stating any vault fact.",
     "inputSchema": {"type": "object", "properties": {
         "claim": {"type": "string"}}, "required": ["claim"]}},
    {"name": "ledger_induce",
     "description": "Aggregate over a fact's full history: number of "
                    "changes, value timeline, trend for numeric values.",
     "inputSchema": {"type": "object", "properties": {
         "entity": {"type": "string"}, "attribute": {"type": "string"}},
         "required": ["entity", "attribute"]}},
    {"name": "ledger_current",
     "description": "List all current facts in the ledger (the full "
                    "snapshot).",
     "inputSchema": {"type": "object", "properties": {}}},
]


def _call(lg, name: str, args: dict):
    if name == "ledger_query":
        return store.query(lg, args["question"])
    if name == "ledger_verify":
        return store.verify(lg, args["claim"])
    if name == "ledger_induce":
        return store.induce(lg, args["entity"], args["attribute"])
    if name == "ledger_current":
        return [{"entity": c.entity, "attribute": c.attribute,
                 "value": c.value, "as_of": store.day_to_iso(c.day),
                 "source": c.episode_id} for c in lg.current_cards()]
    raise ValueError(f"unknown tool {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True)
    args = ap.parse_args()
    lg = store.load(args.vault)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = req.get("id")
        method = req.get("method", "")
        if method == "initialize":
            result = {"protocolVersion":
                      req.get("params", {}).get("protocolVersion",
                                                "2025-03-26"),
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "vault-ledger",
                                     "version": "0.1.0"}}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            p = req.get("params", {})
            try:
                out = _call(lg, p.get("name"), p.get("arguments", {}))
                result = {"content": [{"type": "text",
                                       "text": json.dumps(
                                           out, ensure_ascii=False,
                                           indent=2)}]}
            except Exception as exc:
                result = {"content": [{"type": "text",
                                       "text": f"error: {exc}"}],
                          "isError": True}
        elif rid is None:
            continue  # notification (e.g. notifications/initialized)
        else:
            print(json.dumps({"jsonrpc": "2.0", "id": rid,
                              "error": {"code": -32601,
                                        "message": f"unknown {method}"}}),
                  flush=True)
            continue
        if rid is not None:
            print(json.dumps({"jsonrpc": "2.0", "id": rid,
                              "result": result}, ensure_ascii=False),
                  flush=True)


if __name__ == "__main__":
    main()
