"""Isolated memory slots: one LoRA per entity, fused exactly by rank
concatenation.

For LoRAs sharing base/config, concatenating A along the rank axis and B
along the rank axis gives (x @ A_cat) @ B_cat = sum_i (x @ A_i) @ B_i —
the fusion IS the sum of deltas, computed exactly, with no retraining.
Deleting a slot = re-fusing without it: physical unlearning in O(fusion),
not O(training). This is the mechanism H2 (bounded interference) and H3
(surgical unlearning) call for.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List, Optional

import mlx.core as mx


def fuse_adapters(slot_dirs: List[str], out_dir: str,
                  weight: float = 1.0) -> dict:
    """Fuse mlx_lm LoRA adapters by rank concatenation. All slots must
    share base model, target layers and scale. `weight` scales every
    slot's contribution (fusion-interference knob)."""
    if not slot_dirs:
        raise ValueError("no slots to fuse")
    os.makedirs(out_dir, exist_ok=True)
    configs = []
    weights_per_slot = []
    for d in slot_dirs:
        with open(os.path.join(d, "adapter_config.json")) as f:
            configs.append(json.load(f))
        weights_per_slot.append(
            mx.load(os.path.join(d, "adapters.safetensors")))
    base = configs[0]
    ranks = [c["lora_parameters"]["rank"] for c in configs]
    scales = {c["lora_parameters"]["scale"] for c in configs}
    if len(scales) != 1:
        raise ValueError(f"mixed LoRA scales: {scales}")
    keys = sorted(weights_per_slot[0].keys())
    for w in weights_per_slot[1:]:
        if sorted(w.keys()) != keys:
            raise ValueError("slots target different layers; cannot fuse")

    fused = {}
    for key in keys:
        parts = [w[key] for w in weights_per_slot]
        if key.endswith(".lora_a"):        # [in, r] -> concat on axis 1
            parts = [p * weight for p in parts]
            fused[key] = mx.concatenate(parts, axis=1)
        elif key.endswith(".lora_b"):      # [r, out] -> concat on axis 0
            fused[key] = mx.concatenate(parts, axis=0)
        else:
            raise ValueError(f"unexpected adapter tensor {key}")
    mx.save_safetensors(os.path.join(out_dir, "adapters.safetensors"),
                        fused)
    out_config = dict(base)
    out_config["lora_parameters"] = dict(base["lora_parameters"])
    out_config["lora_parameters"]["rank"] = sum(ranks)
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(out_config, f, indent=2)
    return {"slots": len(slot_dirs), "fused_rank": sum(ranks),
            "out": out_dir}


def slot_iters(n_rows: int, batch_size: int = 2, epochs: int = 12,
               lo: int = 60, hi: int = 240) -> int:
    return max(lo, min(hi, (n_rows * epochs) // batch_size))


from agentlife.harness.backends import MLXBackend
from agentlife.harness.bm25 import BM25Index
from agentlife.harness.parametric_systems import (_run_lora_training,
                                                  _write_dataset)

from .system import CLSLedgerSystem


class CLSSlotsSystem(CLSLedgerSystem):
    """CLS-Ledger with per-entity isolated slots (composition over the
    monolithic adapter). Only _consolidate differs: selected cards are
    grouped per entity, each group trains its own small-rank LoRA, and
    inference uses the exact fusion. slots.json maps slot -> cards ->
    adapter dir: the physical unlearning index."""

    name = "cls-slots"

    def __init__(self, *args, slot_rank: int = 4, slot_min_cards: int = 2,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.slot_rank = slot_rank
        self.slot_min_cards = slot_min_cards

    def _consolidate(self, day: int) -> None:
        self.n_sleeps += 1
        sleep_dir = os.path.join(self.workdir, f"sleep-{self.n_sleeps:02d}")
        os.makedirs(sleep_dir, exist_ok=True)
        selected = self.ledger.select_for_consolidation(
            day, policy=self.policy, budget=self.budget)
        groups: Dict[str, list] = {}
        counts: Dict[str, int] = {}
        for c in selected:
            counts[c.entity] = counts.get(c.entity, 0) + 1
        for c in selected:
            slot = (c.entity if counts[c.entity] >= self.slot_min_cards
                    else "misc")
            groups.setdefault(slot, []).append(c)

        slot_dirs, slot_meta = [], {}
        for slot in sorted(groups):
            cards = groups[slot]
            rows = self._distill_rows(cards, day)
            for r in rows:
                r.pop("_card_id", None)
            sdir = os.path.join(sleep_dir, "slots",
                                slot.replace(" ", "_").lower())
            data_dir = os.path.join(sdir, "data")
            adapter = os.path.join(sdir, "adapter")
            n_valid = max(2, len(rows) // 10)
            _write_dataset(data_dir, rows, rows[-n_valid:])
            iters = slot_iters(len(rows))
            _run_lora_training(self.model_id, data_dir, adapter, iters,
                               mask_prompt=True, lora_rank=self.slot_rank)
            slot_dirs.append(adapter)
            slot_meta[slot] = {"adapter": adapter, "n_rows": len(rows),
                               "iters": iters,
                               "card_ids": [c.card_id for c in cards]}

        fused = os.path.join(sleep_dir, "adapter")
        info = fuse_adapters(slot_dirs, fused)
        with open(os.path.join(sleep_dir, "slots.json"), "w") as f:
            json.dump({"slots": slot_meta, "fusion": info}, f, indent=2)
        self.ledger.dump(os.path.join(sleep_dir, "ledger.jsonl"))
        with open(os.path.join(sleep_dir, "consolidation.json"), "w") as f:
            json.dump({
                "sleep": self.n_sleeps, "day": day,
                "cards_selected": len(selected),
                "n_slots": len(groups),
                "policy": self.policy, "mode": self.mode,
                "slot_rank": self.slot_rank,
                "fused_rank": info["fused_rank"],
                "train_rows": sum(m["n_rows"] for m in slot_meta.values()),
            }, f, indent=2)
        self.consolidated_ids = {c.card_id for c in selected}
        self.all_index = BM25Index()
        self.cards_by_id = {}
        for c in self.ledger.current_cards():
            self.all_index.add(c.card_id, c.text())
            self.cards_by_id[c.card_id] = c
        self.backend = MLXBackend(self.model_id, adapter_path=fused,
                                  cache_dir=self.cache_dir)
        for name in ("consolidation.json", "ledger.jsonl", "slots.json"):
            shutil.copyfile(os.path.join(sleep_dir, name),
                            os.path.join(self.workdir, name))
