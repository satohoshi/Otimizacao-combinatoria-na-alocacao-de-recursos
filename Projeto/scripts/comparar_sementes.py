"""quanto o desempenho do nsga2 muda entre sementes

le a grade (semente 1, results/grade/<cfg>/) e as sementes 2 e 3
(results/grade_sementes/<cfg>_s<n>/) e grava em results/grade_sementes/:
    sementes_metricas.csv   metricas por (config, semente, freq, K), custo 0
    sementes_resumo.csv     por (config, freq): media nos K de cada semente e
                            media, dp e amplitude entre as 3
    sementes_cplex.csv      o mesmo comparado com o cplex (cplex - media das sementes)
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
GRADE = RAIZ / "results" / "grade"
SEM = RAIZ / "results" / "grade_sementes"
CFGS = ["max_sharpe_variancia", "max_sharpe_cvar", "var_min_variancia", "var_min_cvar"]
FREQS = ["sem_rebal", "anual", "semestral", "trimestral", "mensal"]


def ler(p):
    m = pd.read_csv(p / "metricas.csv")
    m = m[(m.tratamento == "degrau_neutralizado") & (m.custo == 0) & m.freq.isin(FREQS)].copy()
    m["K"] = m.K.astype(int)
    return m[["freq", "K", "sharpe_cdi", "retorno_anual", "volatilidade_anual", "drawdown_max"]]


def main():
    linhas = []
    for c in CFGS:
        fontes = {1: GRADE / f"{c}_nsga2"} | {s: SEM / f"{c}_nsga2_s{s}" for s in (2, 3)}
        for s, p in fontes.items():
            if (p / "metricas.csv").exists():
                linhas.append(ler(p).assign(cfg=c, semente=s))
    d = pd.concat(linhas)
    d.to_csv(SEM / "sementes_metricas.csv", index=False)
    por = d.groupby(["cfg", "freq", "semente"]).sharpe_cdi.mean().unstack("semente")
    por.columns = [f"sharpe_s{s}" for s in por.columns]
    cols = list(por.columns)
    por["media"], por["dp"] = por[cols].mean(axis=1), por[cols].std(axis=1, ddof=1)
    por["amplitude"] = por[cols].max(axis=1) - por[cols].min(axis=1)
    por = por.reset_index()
    por.to_csv(SEM / "sementes_resumo.csv", index=False)
    cp = []
    for c in CFGS:
        m = ler(GRADE / f"{c}_cplex").groupby("freq").sharpe_cdi.mean()
        for r in por[por.cfg == c].itertuples():
            cp.append({"cfg": c, "freq": r.freq, "sharpe_cplex": m[r.freq], "sharpe_nsga2_media": r.media,
                       "sharpe_nsga2_dp": r.dp, "dif_cplex_menos_nsga2": m[r.freq] - r.media})
    pd.DataFrame(cp).to_csv(SEM / "sementes_cplex.csv", index=False)
    print(por.round(3).to_string())


if __name__ == "__main__":
    main()
