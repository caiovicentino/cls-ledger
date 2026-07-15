"""Exact, auditable scoring. No LLM judge in the main metric.

Classification per response:
  correct     - contains an accepted value, no rejected value
  ambiguous   - contains both accepted and rejected values
  stale       - contains only a superseded value (freshness query types)
  hallucination - absent query answered with another entity's value
  wrong_value - contains a rejected value on other query types
  abstain     - says 'unknown' on an answerable query
  miss        - none of the above
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Dict, List

from ..banks import UNKNOWN_MARKERS
from ..schema import (QT_ABSENT, QT_CURRENT_UPDATED, QT_CURRENT_VOLATILE,
                      QT_ONLINE, QT_POINT_IN_TIME, QT_REVOKED)

FRESHNESS_QTYPES = {QT_CURRENT_UPDATED, QT_CURRENT_VOLATILE,
                    QT_POINT_IN_TIME, QT_ONLINE, QT_REVOKED}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("’", "'").replace("‘", "'")
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def contains(value: str, text: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(normalize(value)) + r"(?!\w)"
    return re.search(pattern, text) is not None


def classify(response: str, q: dict) -> str:
    text = normalize(response or "")
    if not text:
        return "miss"
    accepted_hit = any(contains(v, text) for v in q["accepted"])
    acc_lower = {normalize(v) for v in q["accepted"]}
    rejected = [v for v in q["rejected"] if normalize(v) not in acc_lower]
    rejected_hit = any(contains(v, text) for v in rejected)
    if accepted_hit and not rejected_hit:
        return "correct"
    if accepted_hit and rejected_hit:
        return "ambiguous"
    if rejected_hit:
        if q["qtype"] in FRESHNESS_QTYPES:
            return "stale"
        if q["qtype"] == QT_ABSENT:
            return "hallucination"
        return "wrong_value"
    if q["qtype"] != QT_ABSENT and any(contains(m, text)
                                       for m in UNKNOWN_MARKERS):
        return "abstain"
    return "miss"


MONTHS_RE = ("January|February|March|April|May|June|July|August|"
             "September|October|November|December")


def check_disposition(rule: str, response: str, accepted) -> "bool | None":
    """Rule-based adherence check on the RAW response (case matters for
    city_upper). Returns None when the response doesn't exhibit the
    construct at all (excluded from the adherence denominator)."""
    if rule == "date_first":
        day_first = re.search(
            r"\b\d{1,2}(st|nd|rd|th)?( of)? (" + MONTHS_RE + ")",
            response, re.I)
        month_first = re.search(
            "(" + MONTHS_RE + r") \d{1,2}(st|nd|rd|th)?\b", response, re.I)
        if day_first and not month_first:
            return True
        if month_first and not day_first:
            return False
        return None
    if rule == "city_upper":
        for v in accepted:
            if v.upper() == v.lower():
                continue
            if v.upper() in response:
                return True
            if re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", response,
                         re.I):
                return False
        return None
    return None


def _bucket(n: int) -> str:
    return {0: "0", 1: "1"}.get(n, "2+")


def score(responses: Dict[str, str], answers: List[dict]) -> dict:
    """responses: query_id -> response text; answers: private query rows."""
    rows = []
    for q in answers:
        if q["query_id"] not in responses:
            continue
        cls = classify(responses[q["query_id"]], q)
        row = {
            "query_id": q["query_id"], "qtype": q["qtype"],
            "heat": q["heat"], "hops": q["hops"],
            "n_updates": _bucket(q["n_updates_before"]),
            "class": cls, "correct": int(cls == "correct"),
            "response": responses[q["query_id"]],
            "answer_display": q["answer_display"],
        }
        if cls == "correct" and q.get("dispositions"):
            row["disposition_checks"] = {
                rule: check_disposition(rule, responses[q["query_id"]],
                                        q["accepted"])
                for rule in q["dispositions"]}
        rows.append(row)

    def acc(subset):
        return (sum(r["correct"] for r in subset) / len(subset)
                if subset else None)

    final = [r for r in rows if r["qtype"] != QT_ONLINE]
    online = [r for r in rows if r["qtype"] == QT_ONLINE]
    by_qtype = {}
    for r in final:
        by_qtype.setdefault(r["qtype"], []).append(r)
    by_heat = {}
    for r in final:
        by_heat.setdefault(r["heat"], []).append(r)
    by_upd = {}
    for r in final:
        by_upd.setdefault(r["n_updates"], []).append(r)
    classes = {}
    for r in rows:
        classes[r["class"]] = classes.get(r["class"], 0) + 1

    adherence = {}
    for r in rows:
        for rule, ok in (r.get("disposition_checks") or {}).items():
            if ok is None:
                continue
            a = adherence.setdefault(rule, {"n": 0, "adherent": 0})
            a["n"] += 1
            a["adherent"] += int(ok)
    for a in adherence.values():
        a["rate"] = a["adherent"] / a["n"] if a["n"] else None

    return {
        "n_final": len(final),
        "disposition_adherence": adherence,
        "n_online": len(online),
        "accuracy_final": acc(final),
        "accuracy_online": acc(online),
        "by_qtype": {k: {"n": len(v), "acc": acc(v)}
                     for k, v in sorted(by_qtype.items())},
        "by_heat": {k: {"n": len(v), "acc": acc(v)}
                    for k, v in sorted(by_heat.items())},
        "by_n_updates": {k: {"n": len(v), "acc": acc(v)}
                         for k, v in sorted(by_upd.items())},
        "error_classes": classes,
        "rows": rows,
    }


def print_report(report: dict) -> None:
    def pct(x):
        return "  n/a" if x is None else f"{100 * x:5.1f}%"

    print(f"final queries : {report['n_final']:4d}  "
          f"acc {pct(report['accuracy_final'])}")
    print(f"online queries: {report['n_online']:4d}  "
          f"acc {pct(report['accuracy_online'])}")
    print("\nby query type:")
    for k, v in report["by_qtype"].items():
        print(f"  {k:<18} n={v['n']:<4} acc {pct(v['acc'])}")
    print("\nby heat (final):")
    for k, v in report["by_heat"].items():
        print(f"  {k:<6} n={v['n']:<4} acc {pct(v['acc'])}")
    print("\nby #updates before ask (final):")
    for k, v in report["by_n_updates"].items():
        print(f"  {k:<3} n={v['n']:<4} acc {pct(v['acc'])}")
    if report.get("disposition_adherence"):
        print("\ndisposition adherence (rule-based, correct answers only):")
        for k, v in report["disposition_adherence"].items():
            print(f"  {k:<12} {v['adherent']}/{v['n']} "
                  f"({100*(v['rate'] or 0):.0f}%)")
    print("\nresponse classes (all):")
    for k, v in sorted(report["error_classes"].items()):
        print(f"  {k:<14} {v}")
