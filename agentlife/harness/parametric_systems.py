"""Parametric baselines: memory in the WEIGHTS, not in the context.

- lora : naive continual LoRA. Trains next-token on the raw episode texts
  once at end of life, then answers with NO notes in context. The expected
  (and interesting) failure: knowledge stated as prose does not become
  queryable knowledge without augmentation.
- seal : SEAL-lite. Converts each informative episode into synthetic QA
  pairs (the "self-edit" of Self-Adapting LLMs, simplified: the editor is
  an external cheap LLM instead of the model itself), trains LoRA on the
  QAs, answers with no notes in context.

Both only consolidate at end of life (finalize); mid-life online queries
are answered 'unknown'. That is itself a measurement: a system that only
consolidates at the end cannot answer during the stream.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import List, Optional

from .backends import LLMBackend, MLXBackend, OpenAIBackend
from .systems import MemorySystem

PARAMETRIC_SYSTEM_PROMPT = (
    "You are a personal assistant. You have memorized the user's life "
    "notes (dated 'Day N'). Answer the question from memory. Be concise: "
    "answer with the value only (a name, a date, a city, a code), no "
    "explanations. If you never learned the information, answer 'unknown'."
)

QA_GEN_PROMPT = (
    "Note from a personal assistant's diary (day {day}):\n{text}\n\n"
    "Write {n} question-answer pairs that a user might later ask their "
    "assistant about the FACTS in this note. Rules:\n"
    "- questions must be self-contained: use full names, and mention "
    "'as of day {day}' when the fact could change over time;\n"
    "- answers must be the short value only (a name, date, city, code);\n"
    "- only ask about facts actually in the note.\n"
    "Return ONLY a JSON array like "
    '[{{"q": "...", "a": "..."}}, ...] with no other text.'
)


def _run_lora_training(model_id: str, data_dir: str, adapter_path: str,
                       iters: int, batch_size: int = 2,
                       num_layers: int = 8, learning_rate: float = 1e-4,
                       mask_prompt: bool = False, seed: int = 0,
                       lora_rank: Optional[int] = None) -> None:
    cmd = [sys.executable, "-m", "mlx_lm", "lora",
           "--model", model_id, "--train",
           "--data", data_dir,
           "--adapter-path", adapter_path,
           "--iters", str(iters),
           "--batch-size", str(batch_size),
           "--num-layers", str(num_layers),
           "--learning-rate", str(learning_rate),
           "--max-seq-length", "512",
           "--save-every", str(iters),
           "--steps-per-report", "50",
           "--seed", str(seed)]
    if lora_rank is not None:
        os.makedirs(adapter_path, exist_ok=True)
        cfg = os.path.join(adapter_path, "train_config.yaml")
        with open(cfg, "w") as f:
            f.write("lora_parameters:\n"
                    f"  rank: {lora_rank}\n"
                    "  scale: 20.0\n"
                    "  dropout: 0.0\n")
        cmd += ["-c", cfg]
    if mask_prompt:
        cmd.append("--mask-prompt")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"LoRA training failed:\n{result.stdout[-2000:]}\n"
            f"{result.stderr[-2000:]}")


def _write_dataset(data_dir: str, train_rows: List[dict],
                   valid_rows: List[dict]) -> None:
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "train.jsonl"), "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(data_dir, "valid.jsonl"), "w") as f:
        for r in valid_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class LoRANaiveSystem(MemorySystem):
    name = "lora"

    def __init__(self, model_id: str, workdir: str, iters: int = 400,
                 cache_dir: Optional[str] = None):
        self.model_id = model_id
        self.workdir = workdir
        self.iters = iters
        self.cache_dir = cache_dir
        self.episodes: List[dict] = []
        self.backend: Optional[LLMBackend] = None

    def ingest(self, episode: dict) -> None:
        self.episodes.append(episode)

    def _train_rows(self) -> List[dict]:
        return [{"text": e["text"]} for e in self.episodes]

    def finalize(self) -> None:
        rows = self._train_rows()
        n_valid = max(2, len(rows) // 10)
        data_dir = os.path.join(self.workdir, "data")
        adapter = os.path.join(self.workdir, "adapter")
        _write_dataset(data_dir, rows, rows[-n_valid:])
        _run_lora_training(self.model_id, data_dir, adapter, self.iters,
                           mask_prompt=False)
        self.backend = MLXBackend(self.model_id, adapter_path=adapter,
                                  cache_dir=self.cache_dir)

    def answer(self, query: dict) -> str:
        if self.backend is None:
            return "unknown"
        user = (f"QUESTION (asked on day {query['day_asked']}): "
                f"{query['question']}\nAnswer:")
        return self.backend.complete(PARAMETRIC_SYSTEM_PROMPT, user)


class SEALLiteSystem(LoRANaiveSystem):
    name = "seal"

    def __init__(self, model_id: str, workdir: str, iters: int = 600,
                 cache_dir: Optional[str] = None,
                 editor_model: str = "gpt-4.1-mini", qa_per_episode: int = 3):
        super().__init__(model_id, workdir, iters, cache_dir)
        self.editor = OpenAIBackend(editor_model, cache_dir=cache_dir)
        self.qa_per_episode = qa_per_episode

    def _gen_qas(self, episode: dict) -> List[dict]:
        m = re.match(r"Day (\d+):", episode["text"])
        day = m.group(1) if m else "?"
        prompt = QA_GEN_PROMPT.format(day=day, text=episode["text"],
                                      n=self.qa_per_episode)
        raw = self.editor.complete(
            "You convert notes into training QA pairs. Output only JSON.",
            prompt, max_tokens=400)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.M).strip()
        try:
            pairs = json.loads(raw)
        except json.JSONDecodeError:
            return []
        out = []
        for p in pairs:
            if isinstance(p, dict) and p.get("q") and p.get("a"):
                out.append({"q": str(p["q"]), "a": str(p["a"])})
        return out

    def _train_rows(self) -> List[dict]:
        rows = []
        for e in self.episodes:
            if not e.get("fact_keys"):
                continue  # noise episodes carry no facts worth memorizing
            for qa in self._gen_qas(e):
                rows.append({"messages": [
                    {"role": "system", "content": PARAMETRIC_SYSTEM_PROMPT},
                    {"role": "user", "content": qa["q"]},
                    {"role": "assistant", "content": qa["a"]},
                ]})
        return rows

    def finalize(self) -> None:
        rows = self._train_rows()
        if not rows:
            raise RuntimeError("no QA pairs generated")
        n_valid = max(2, len(rows) // 10)
        data_dir = os.path.join(self.workdir, "data")
        adapter = os.path.join(self.workdir, "adapter")
        _write_dataset(data_dir, rows, rows[-n_valid:])
        _run_lora_training(self.model_id, data_dir, adapter, self.iters,
                           mask_prompt=True)
        self.backend = MLXBackend(self.model_id, adapter_path=adapter,
                                  cache_dir=self.cache_dir)
