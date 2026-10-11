"""Análise do piloto pressão: tabela (m × ativ) + veredito (i)-(iv), SEM amaciar.

Lê results/{rot}-m{m}-s{seed}.jsonl.gz (todos os seeds presentes), agrega por
célula (m, ativ) e escreve pressure-results.md. Regra da casa: números reais,
zero teatro; se (iii) falhar (relu ressuscita morto-real), destaca como RED FLAG.
"""
import glob
import gzip
import json
import os

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "results")

ROTS = ["softplus-b300", "softplus-b30", "relu"]
MS = [4, 8, 16]


def carrega():
    runs = []
    for p in sorted(glob.glob(os.path.join(OUT, "*.jsonl.gz"))):
        d = json.load(gzip.open(p, "rt"))
        runs.append(d)
    return runs


def cel(runs, rot, m):
    return [r for r in runs if r["rot"] == rot and r["m"] == m]


def agg(rs, chave):
    return [r[chave] for r in rs]


def _durstat(rs, campo):
    meds = [r[campo]["med"] for r in rs if r[campo]["med"] is not None]
    p90s = [r[campo]["p90"] for r in rs if r[campo]["p90"] is not None]
    med = float(np.median(meds)) if meds else None
    p90 = float(np.median(p90s)) if p90s else None
    return med, p90


def _mortes(rs):
    """A partir dos episódios v3 (completos+censurados) de cada run da célula:
    separa morte-por-init (t_morte==0, 'born dead') de morte-DURANTE-treino
    (t_morte>0) — só a última é o objeto de T2'. Retorna contagens e z_min das
    mortes-durante-treino (quão fundo a margem-líder mergulhou)."""
    born = durante = 0
    zmins_durante = []
    for r in rs:
        for e in r.get("v3_completos", []) + r.get("v3_censurados", []):
            if e["t_morte"] == 0:
                born += 1
            else:
                durante += 1
                zmins_durante.append(e["z_min"])
    zmin_menos_fundo = max(zmins_durante) if zmins_durante else None  # mais raso
    zmin_mais_fundo = min(zmins_durante) if zmins_durante else None
    return dict(born=born, durante=durante,
                zmin_raso=zmin_menos_fundo, zmin_fundo=zmin_mais_fundo)


def main():
    runs = carrega()
    if not runs:
        print("sem resultados ainda.")
        return
    linhas = []
    linhas.append(f"# Piloto pressão (etapa-4a.3) — tabela e veredito\n")
    linhas.append(f"_Runs carregados: {len(runs)} (esperado 36). "
                  f"T=0, η=3e-4, H=3e5, n=64, alvo rico, seeds 42-45, n_σ=10._\n")

    # ---- tabela principal (uma linha por célula m×ativ, agregada nos seeds) ----
    linhas.append("## Tabela (m × ativ), agregado nos seeds presentes\n")
    linhas.append("| ativ | m | seeds | loss_plato (min–max) | Σmortos_real | "
                  "Σrev_real (neur) | Σv2_comp | Σv3_comp | Σv3_zmin<0 | "
                  "v3 dur med/p90 |")
    linhas.append("|---|--:|--:|---|--:|--:|--:|--:|--:|---|")
    tabela = {}
    for rot in ROTS:
        for m in MS:
            rs = cel(runs, rot, m)
            if not rs:
                continue
            lp = agg(rs, "loss_plato")
            mortos = sum(agg(rs, "n_mortos_real"))
            rev = sum(agg(rs, "n_revive_real"))
            revn = sum(agg(rs, "n_neuronios_revive_real"))
            v2c = sum(agg(rs, "v2_n_completos"))
            v3c = sum(agg(rs, "v3_n_completos"))
            v3zn = sum(agg(rs, "v3_n_completos_zmin_neg"))
            med, p90 = _durstat(rs, "v3_dur")
            mrt = _mortes(rs)
            tabela[(rot, m)] = dict(
                nseed=len(rs), lp_min=min(lp), lp_max=max(lp), lp_mean=np.mean(lp),
                mortos=mortos, rev=rev, revn=revn, v2c=v2c, v3c=v3c, v3zn=v3zn,
                dur_med=med, dur_p90=p90, **mrt)
            dstr = (f"{med:.0f}/{p90:.0f}" if med is not None else "—")
            linhas.append(
                f"| {rot} | {m} | {len(rs)} | {min(lp):.4f}–{max(lp):.4f} | "
                f"{mortos} | {rev} ({revn}) | {v2c} | {v3c} | {v3zn} | {dstr} |")
    linhas.append("")

    # ---- tabela de mortes (born-dead vs durante-treino) ----
    linhas.append("## Mortes: born-dead (t_morte=0) vs durante-treino (t_morte>0)\n")
    linhas.append("_Só a morte DURANTE-treino é o objeto de T2' (neurônio vivo que "
                  "morre). z_min raso/fundo = a mais rasa e a mais funda das mortes "
                  "durante-treino da célula._\n")
    linhas.append("| ativ | m | Σborn-dead | Σdurante-treino | z_min raso | z_min fundo | Σrev_real |")
    linhas.append("|---|--:|--:|--:|--:|--:|--:|")
    for rot in ROTS:
        for m in MS:
            if (rot, m) not in tabela:
                continue
            c = tabela[(rot, m)]
            zr = f"{c['zmin_raso']:.4f}" if c["zmin_raso"] is not None else "—"
            zf = f"{c['zmin_fundo']:.4f}" if c["zmin_fundo"] is not None else "—"
            linhas.append(f"| {rot} | {m} | {c['born']} | {c['durante']} | "
                          f"{zr} | {zf} | {c['rev']} |")
    linhas.append("")

    # ---- veredito ----
    linhas.append("## Veredito (i)-(iv) — sem amaciar\n")

    # (i) loss platôia > 0
    todos_lp = agg(runs, "loss_plato")
    lp_min = min(todos_lp)
    # tendência com m: para cada ativ, loss_plato médio por m
    linhas.append("**(i) loss platôia em valor > 0 (pressão persiste):**")
    piso = 1e-3
    ok_i = lp_min > piso
    linhas.append(f"- loss_plato mínimo sobre todos os runs = {lp_min:.4e} "
                  f"(piso de 'pressão' arbitrado a {piso:.0e}). "
                  f"{'CONFIRMA' if ok_i else 'FALHA (algum run zerou)'}.")
    for rot in ROTS:
        por_m = []
        for m in MS:
            if (rot, m) in tabela:
                por_m.append(f"m{m}={tabela[(rot, m)]['lp_mean']:.4f}")
        if por_m:
            linhas.append(f"  - {rot}: " + ", ".join(por_m)
                          + "  (esperado: cresce quando m cai)")

    # (ii) softplus resurrections genuine
    linhas.append("\n**(ii) softplus tem ressurreições genuínas (>0, dur física):**")
    for rot in ("softplus-b300", "softplus-b30"):
        rev_tot = sum(tabela[(rot, m)]["rev"] for m in MS if (rot, m) in tabela)
        v3_tot = sum(tabela[(rot, m)]["v3c"] for m in MS if (rot, m) in tabela)
        cel_rev = {m: tabela[(rot, m)]["rev"] for m in MS if (rot, m) in tabela}
        linhas.append(f"- {rot}: Σrev_real={rev_tot} (por m: {cel_rev}), "
                      f"Σv3_completos={v3_tot}.")
    # deaths during-training para softplus (o objeto real de T2')
    sp_dur = sum(tabela[(rot, m)]["durante"]
                 for rot in ("softplus-b300", "softplus-b30")
                 for m in MS if (rot, m) in tabela)
    sp_zraso = [tabela[(rot, m)]["zmin_raso"]
                for rot in ("softplus-b300", "softplus-b30")
                for m in MS if (rot, m) in tabela
                and tabela[(rot, m)]["zmin_raso"] is not None]
    raso = max(sp_zraso) if sp_zraso else None
    linhas.append(f"- softplus tem {sp_dur} mortes DURANTE-treino (t_morte>0) no total "
                  f"(alvo real de T2'); a mais rasa mergulhou a z_min={raso} "
                  f"(margem-líder negativa em TODOS os dados).")
    ok_ii = any(sum(tabela[(rot, m)]["rev"] for m in MS if (rot, m) in tabela) > 0
                for rot in ("softplus-b300", "softplus-b30"))
    if ok_ii:
        linhas.append("- CONFIRMA: softplus ressuscita (Σrev_real>0 acima).")
    else:
        linhas.append("- **FALHA**: softplus NÃO ressuscita (morto-real->vivo = 0) "
                      "mesmo com pressão persistente (loss-platô>0) e mortes ocorrendo. "
                      "Mecanismo: a T=0 (determinístico) a margem-líder mergulha FUNDO "
                      "(z_min≪0), onde σ(βz) é desprezível ⟹ sem gradiente de retorno e "
                      "sem ruído p/ escapar ⟹ morte TERMINAL. É a condição-que-MATA "
                      "a hipótese descrita no pré-desenho.")

    # (iii) relu exactly 0 resurrections of truly-dead
    linhas.append("\n**(iii) relu: EXATAMENTE 0 ressurreições de morto-real "
                  "(grad≡0):**")
    relu_rev = sum(tabela[("relu", m)]["rev"] for m in MS if ("relu", m) in tabela)
    relu_zn = sum(tabela[("relu", m)]["v3zn"] for m in MS if ("relu", m) in tabela)
    relu_v3 = sum(tabela[("relu", m)]["v3c"] for m in MS if ("relu", m) in tabela)
    linhas.append(f"- ground-truth Σrev_real(relu) = {relu_rev} "
                  f"(morto-real zmax<0 -> vivo zmax>0).")
    linhas.append(f"- régua v3 Σzmin<0 completos(relu) = {relu_zn} "
                  f"(episódios v3 que ressuscitaram tendo mergulhado <0).")
    linhas.append(f"- régua v3 Σcompletos(relu) = {relu_v3} "
                  f"(inclui rasos: 0<margem<lim_morte, neurônio NUNCA morto-real).")
    ok_iii = (relu_rev == 0 and relu_zn == 0)
    if ok_iii:
        linhas.append("- CONFIRMA: relu não ressuscita morto-real. "
                      "Episódios v3 de relu (se houver) são de margem-líder rasa "
                      "(nunca <0), i.e. o limiar da régua > 0 pegou quase-mortos.")
    else:
        linhas.append("- **RED FLAG**: relu ressuscitou morto-real. Investigar "
                      "(bug de régua/harness OU margem-líder não-morta-em-todos). "
                      "Ver revive_real_eventos no gz do run.")

    # (iv) beta smaller -> more resurrections
    linhas.append("\n**(iv) β menor ⟹ mais ressurreições (β30 vs β300):**")
    for m in MS:
        a = tabela.get(("softplus-b300", m))
        b = tabela.get(("softplus-b30", m))
        if a and b:
            linhas.append(f"- m={m}: β300 rev_real={a['rev']} v3={a['v3c']}  |  "
                          f"β30 rev_real={b['rev']} v3={b['v3c']}")
    rev300 = sum(tabela[("softplus-b300", m)]["rev"] for m in MS
                 if ("softplus-b300", m) in tabela)
    rev30 = sum(tabela[("softplus-b30", m)]["rev"] for m in MS
                if ("softplus-b30", m) in tabela)
    linhas.append(f"- total: β300 Σrev_real={rev300}  vs  β30 Σrev_real={rev30}. "
                  f"{'direção CONFIRMA (β30>β300)' if rev30 > rev300 else 'direção NÃO confirma'}"
                  " (exploratório, sem estatística).")

    txt = "\n".join(linhas) + "\n"
    with open(os.path.join(AQUI, "pressure-results.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
