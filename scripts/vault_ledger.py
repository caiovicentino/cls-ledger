"""CLS-Ledger applied to Caio's Obsidian vault (cerebro/major).

Extraction was performed by Claude reading the active notes directly
(2026-07-15) — no external API touched the vault content. Cards carry
real ISO dates (stored as ordinal days) and per-note provenance.

Builds: Memory/_ledger/vault_ledger.jsonl + VAULT_LEDGER.md (readable
snapshot: current facts, superseded history, induction answers).

Usage: PYTHONPATH=. python3 scripts/vault_ledger.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date

sys.path.insert(0, os.getcwd())
from clsledger.ledger import Card, Ledger  # noqa: E402

VAULT = "/Volumes/SSD Major/cerebro/major"
OUT = os.path.join(VAULT, "Memory", "_ledger")


def d(iso: str) -> int:
    return date.fromisoformat(iso).toordinal()


# (entity, attribute, value, iso_date, source_note)
CARDS = [
    # ---- arco WANDERING / papers (Zenodo, OpenInterpretability)
    ("Paper #1 Tool-Entropy Collapse", "doi", "10.5281/zenodo.20368601",
     "2026-05-24", "fish/reference_openinterp_research_state"),
    ("Paper #2 Right Locus", "doi", "10.5281/zenodo.20490278",
     "2026-05-27", "fish/reference_openinterp_research_state"),
    ("Paper #3 Multi-Channel Signatures", "doi", "10.5281/zenodo.20490284",
     "2026-05-28", "fish/reference_openinterp_research_state"),
    ("Paper #4 Modality Matters", "doi", "10.5281/zenodo.20490286",
     "2026-05-31", "fish/reference_openinterp_research_state"),
    ("Paper #4 Modality Matters", "finding",
     "behavioral interruption rescues finalization 30->70% "
     "(McNemar p=0.021)", "2026-05-31", "_reports/2026-05"),
    ("Companion context-rot", "doi", "10.5281/zenodo.20500053",
     "2026-06-01", "fish/reference_openinterp_research_state"),
    ("Paper #6 The Lever Is Late", "doi", "10.5281/zenodo.20534219",
     "2026-06-03", "fish/reference_openinterp_research_state"),
    ("Paper #16 Criterion Cannot See", "doi", "10.5281/zenodo.21175759",
     "2026-07-03", "claude-call/zenodo-openinterpretability-papers"),
    ("CLS-Ledger paper", "doi", "10.5281/zenodo.21375429",
     "2026-07-14", "orq/cls-ledger"),
    ("CLS-Ledger paper", "doi", "10.5281/zenodo.21384691",
     "2026-07-15", "orq/cls-ledger"),
    ("CLS-Ledger paper", "concept_doi", "10.5281/zenodo.21375428",
     "2026-07-14", "orq/cls-ledger"),
    ("paper series", "count_published", "16 papers ate 2026-07-04",
     "2026-07-04", "claude-call/zenodo-openinterpretability-papers"),
    ("ICML 2026 MI Workshop", "status",
     "poster #73 aceito (Hallucination-Induction, Not Calibration) — "
     "primeira aceitacao peer-reviewed", "2026-06-15", "_reports/2026-06"),

    # ---- regras do X (supersedencias REAIS documentadas)
    ("regra de cards", "valor",
     "clawd_card.py padrao Apple (Inter font + gradient)",
     "2026-04-14", "clawd/feedback_card_apple_standard"),
    ("regra de cards", "valor",
     "NAO criar cards; sem source media -> text-only",
     "2026-04-15", "clawd/feedback_no_clawd_card"),
    ("regra de media", "valor", "todo post deve ter media (v1)",
     "2026-04-12", "clawd/feedback_media_always"),
    ("regra de media", "valor",
     "todo post deve ter media do tweet-fonte via grab-media (v2)",
     "2026-04-14", "clawd/feedback_media_always_v2"),
    ("crypto macro track", "status", "KILL — nao curar (15L vs 95L)",
     "2026-04-15", "clawd/feedback_kill_crypto_macro_track"),
    ("regra de links", "valor",
     "nao embedar URL de source nos posts (algoritmo penaliza)",
     "2026-05-08", "clawd/feedback_no_source_link"),

    # ---- naming / estrategia (pivôs empilhados)
    ("weight-quant work", "nome", "PolarQuant",
     "2026-04-01", "fish (era pre-rebrand)"),
    ("weight-quant work", "nome",
     "HLWQ (PolarQuant e do Google, Han et al. 2502.02617)",
     "2026-04-20", "clawd/project_hlwq_rebrand_notice"),
    ("estrategia OpenInterp", "valor", "AgentGuard SaaS como via",
     "2026-05-23", "fish/project_openinterp_capstone_pivot"),
    ("estrategia OpenInterp", "valor", "research-not-SaaS",
     "2026-06-14", "_reports/2026-06"),
    ("estrategia OpenInterp", "valor",
     "interp-as-AUDIT, not interp-as-control",
     "2026-06-18", "_reports/2026-06"),

    # ---- series temporais (inducao real)
    ("pasta fish", "file_count", "189", "2026-04-30", "_reports/2026-05"),
    ("pasta fish", "file_count", "319", "2026-06-01", "_reports/2026-05"),
    ("pasta fish", "file_count", "345", "2026-07-01", "_reports/2026-06"),
    ("conta @0xCVYH", "baseline_followers", "17500",
     "2026-04-17", "clawd/project_x_metas_loop"),

    # ---- issues persistentes
    ("arxiv 2603.29078", "status", "404 — open issue (confirmar ID)",
     "2026-06-01", "_reports/2026-05"),
    ("arxiv 2603.29078", "status",
     "404 — ainda aberto (2a auditoria consecutiva)",
     "2026-07-01", "_reports/2026-06"),
    ("proposta fish/_archive", "status", "proposta, nao executada",
     "2026-06-01", "_reports/2026-05"),
    ("proposta fish/_archive", "status",
     "well overdue (2a auditoria), aguarda OK do Caio",
     "2026-07-01", "_reports/2026-06"),

    # ---- identidade / usuario
    ("Major", "identidade",
     "assistente 🎖️ do Caio; proativo; seguranca; pt-BR",
     "2026-04-10", "openclaw/IDENTITY"),
    ("Caio Vicentino", "localizacao", "Ceara, Brasil (GMT-3)",
     "2026-04-10", "openclaw/USER"),
    ("Caio Vicentino", "titulo",
     "MakerDAO Ambassador Brazil; DeFi Expert (Ivan on Tech); pioneiro "
     "yield farming BR", "2026-04-10", "openclaw/USER"),
    ("Caio Vicentino", "orcid", "0009-0003-4331-6259",
     "2026-05-19", "fish/reference_caio_orcid"),
    ("Caio Vicentino", "hf_namespace", "caiovicentino1",
     "2026-04-20", "clawd/reference_hf_polarquant_url"),
    ("Caio Vicentino", "lema", "Do Ceara pro Mundo",
     "2026-04-10", "openclaw/USER"),

    # ---- produtos / infra
    ("FabricationGuard", "status", "lancado",
     "2026-04-27", "fish/project_fabricationguard_launch"),
    ("AgentGuard", "versao", "v0.1.0 (4-layer action firewall)",
     "2026-06-10", "_reports/2026-06"),
    ("decision-locator", "status", "shipped (GitHub OpenInterpretability)",
     "2026-06-05", "fish/reference_openinterp_research_state"),
    ("inspect_evals PR #1716", "status", "CI verde, aguarda merge (UK AISI)",
     "2026-06-05", "fish/reference_openinterp_research_state"),
    ("MATS Autumn", "deadline", "2026-06-07",
     "2026-06-01", "fish/reference_openinterp_research_state"),
]


def main() -> None:
    lg = Ledger()
    for ent, attr, val, iso, src in CARDS:
        lg.add(Card(ent, attr, val, d(iso), src))
    os.makedirs(OUT, exist_ok=True)
    lg.dump(os.path.join(OUT, "vault_ledger.jsonl"))

    lines = ["# Vault Ledger — snapshot simbólico do cerebro/major",
             "", f"Gerado por Claude em 2026-07-15 a partir da leitura "
             f"direta das notas ativas. {len(lg.cards)} cards, "
             f"{len(lg.current_cards())} fatos atuais. Proveniência em "
             "cada linha; histórico completo no vault_ledger.jsonl.", "",
             "## Fatos atuais", ""]
    for c in sorted(lg.current_cards(), key=lambda c: (c.entity,
                                                       c.attribute)):
        when = date.fromordinal(c.day).isoformat()
        lines.append(f"- **{c.entity}** · {c.attribute} = {c.value} "
                     f"({when}; fonte: {c.episode_id})")
    lines += ["", "## Supersedidos (história preservada)", ""]
    for c in lg.cards:
        if c.superseded_by:
            when = date.fromordinal(c.day).isoformat()
            lines.append(f"- ~~{c.entity} · {c.attribute} = {c.value}~~ "
                         f"({when}) → superseded")
    lines += ["", "## Induções (agregação simbólica)", ""]

    def hist(ent, attr):
        from clsledger.ledger import norm_key
        return lg.history(f"{norm_key(ent)}.{norm_key(attr)}")

    piv = hist("estrategia OpenInterp", "valor")
    lines.append(f"- A estratégia do OpenInterp mudou **{len(piv)-1} "
                 f"vezes** ({' → '.join(v.value.split(',')[0].split('(')[0].strip() for v in piv)})")
    fc = hist("pasta fish", "file_count")
    first, last = int(fc[0].value), int(fc[-1].value)
    lines.append(f"- fish/ cresceu de {first} para {last} arquivos "
                 f"(**{'increased' if last > first else 'decreased'}**, "
                 f"{len(fc)} medições)")
    cards_r = hist("regra de cards", "valor")
    lines.append(f"- A regra de cards mudou {len(cards_r)-1} vez — ativa: "
                 f"\"{cards_r[-1].value}\"")
    arx = hist("arxiv 2603.29078", "status")
    lines.append(f"- Issue arXiv 2603.29078: persiste há {len(arx)} "
                 f"auditorias mensais consecutivas — segue ABERTA")
    with open(os.path.join(OUT, "VAULT_LEDGER.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"ledger: {len(lg.cards)} cards -> {OUT}")
    print("\n".join(lines[-4:]))


if __name__ == "__main__":
    main()
