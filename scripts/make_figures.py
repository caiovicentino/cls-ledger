"""Generate the paper's 3 primary figures from reports/ artifacts.

Usage: PYTHONPATH=. python3 scripts/make_figures.py
Outputs: paper/fig1_scale.png, paper/fig2_guarantees.png,
         paper/fig3_paraphrase.png (200 dpi)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLS = "#2563eb"
RAG = "#9ca3af"
ACCENT = "#dc2626"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

# ---- Fig 1: scale curve (unseen seeds; L single-seed marked open) ----
days = [30, 90, 365]
cls_mean, cls_sd = [64.6, 71.1, 80.5], [6.7, 1.7, 0]
rag_mean, rag_sd = [50.1, 42.8, 33.9], [4.1, 2.6, 0]

fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.errorbar(days[:2], cls_mean[:2], yerr=cls_sd[:2], color=CLS, marker="o",
            capsize=4, lw=2, label="CLS-Ledger (slots-routed)")
ax.plot(days[1:], cls_mean[1:], color=CLS, lw=2, ls="--")
ax.plot(days[2], cls_mean[2], marker="o", mfc="white", mec=CLS, mew=2,
        ms=8, ls="none")
ax.errorbar(days[:2], rag_mean[:2], yerr=rag_sd[:2], color=RAG, marker="s",
            capsize=4, lw=2, label="BM25 RAG (same reader)")
ax.plot(days[1:], rag_mean[1:], color=RAG, lw=2, ls="--")
ax.plot(days[2], rag_mean[2], marker="s", mfc="white", mec=RAG, mew=2,
        ms=8, ls="none")
ax.annotate("single seed", (365, 80.5), textcoords="offset points",
            xytext=(-8, 10), fontsize=8, color=CLS, ha="right")
ax.set_xscale("log")
ax.set_xticks(days)
ax.set_xticklabels(["30", "90", "365"])
ax.set_xlabel("life length (days)")
ax.set_ylabel("final-exam accuracy (%)")
ax.set_ylim(0, 100)
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig("paper/fig1_scale.png", dpi=200)

# ---- Fig 2: guarantees (probe panel + unlearning panel) ----
fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.2),
                           gridspec_kw={"width_ratios": [1.5, 1]})
names = ["Base model", "Slots (at rest)", "Slots (one active)",
         "Naive LoRA", "SEAL-lite", "CLS monolithic", "Fused slots"]
vals = [93.8, 93.8, 93.8, 87.5, 87.5, 75.0, 0.0]
colors = ["#111827", CLS, CLS, RAG, RAG, RAG, ACCENT]
y = range(len(names))[::-1]
a.barh(list(y), vals, color=colors, height=0.62)
for yi, v in zip(y, vals):
    a.text(v + 1.2, yi, f"{v:.1f}", va="center", fontsize=8.5)
a.set_yticks(list(y))
a.set_yticklabels(names, fontsize=8.5)
a.set_xlim(0, 104)
a.set_xlabel("general-capability probe (%)")
a.set_title("(a) Forgetting", fontsize=10, loc="left")

b.bar([0, 1], [0, 19], color=[CLS, RAG], width=0.55)
b.set_xticks([0, 1])
b.set_xticklabels(["slot drop\n(no retraining)", "re-distillation"],
                  fontsize=8.5)
b.set_ylabel("unrelated answers changed (of 40)")
b.set_ylim(0, 22)
b.text(0, 0.5, "0", ha="center", fontsize=11, fontweight="bold", color=CLS)
b.text(1, 19.4, "19", ha="center", fontsize=11, fontweight="bold")
b.set_title("(b) Unlearning collateral", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig("paper/fig2_guarantees.png", dpi=200)

# ---- Fig 3: template vs paraphrase decomposition (S-1) ----
systems = ["CLS\n(lexical parser)", "CLS\n(semantic parser)",
           "Embeddings RAG", "BM25 RAG"]
template = [88.6, 93.2, 54.5, 52.3]
para = [56.8, 75.0, 63.6, 52.3]
x = range(len(systems))
w = 0.36
fig, ax = plt.subplots(figsize=(5.8, 3.3))
ax.bar([i - w / 2 for i in x], template, w, color=CLS, label="template")
ax.bar([i + w / 2 for i in x], para, w, color="#93c5fd",
       label="paraphrased")
for i in x:
    ax.text(i - w / 2, template[i] + 1, f"{template[i]:.0f}", ha="center",
            fontsize=8.5)
    ax.text(i + w / 2, para[i] + 1, f"{para[i]:.0f}", ha="center",
            fontsize=8.5)
ax.set_xticks(list(x))
ax.set_xticklabels(systems, fontsize=8.5)
ax.set_ylabel("final-exam accuracy (%)")
ax.set_ylim(0, 104)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("paper/fig3_paraphrase.png", dpi=200)
print("3 figures written to paper/")
