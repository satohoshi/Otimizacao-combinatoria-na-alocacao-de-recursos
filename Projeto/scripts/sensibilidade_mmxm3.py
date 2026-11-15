"""sensibilidade ao episodio da MMXM3 de out/2020 a jun/2021

a MMXM3 (em recuperacao judicial) entra nas carteiras de risco minimo com peso
pequeno e a partir de out/2020 tem uma alta especulativa com negociacao de verdade
(~40x a mediana de negocios), o q infla a vol do risco minimo no semestral,
trimestral e mensal. aqui recalcula as metricas sem 2020-10-01 a 2021-06-30 (pra
todas as configs e pro cdi) so pra ver o peso disso. n eh a metrica oficial, o
sharpe aqui eh media do excesso diario / desvio * raiz(252)

saida: results/grade/comparacao/sensibilidade_mmxm3.md e .csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
GRADE = RAIZ / "results" / "grade"
DEST = GRADE / "comparacao"
INI, FIM = "2020-10-01", "2021-06-30"
FREQS = ["sem_rebal", "anual", "semestral", "trimestral", "mensal"]
REGRAS = {"max_sharpe": "Máximo Sharpe", "var_min": "Risco mínimo"}
MET = {"cplex": "CPLEX", "nsga2": "NSGA-II"}
MED = {"variancia": "variância", "cvar": "CVaR"}


def sharpe(e):
    return e.mean() / e.std() * np.sqrt(252)


def main():
    linhas = []
    for r in REGRAS:
        for m in MET:
            for e in MED:
                v = pd.read_csv(GRADE / f"{r}_{e}_{m}" / "valor_diario.csv", index_col="data", parse_dates=True)
                cdi = v["CDI"].pct_change()
                for f in FREQS:
                    for K in (3, 4, 5, 9, 15, 20, 30):
                        col = f"K{K}_{f}_c0.00%"
                        if col not in v:
                            continue
                        ret = v[col].pct_change().dropna()
                        ex = (ret - cdi).dropna()
                        fora = ~((ex.index >= INI) & (ex.index <= FIM))
                        linhas.append({"regra": r, "metodo": m, "medida": e, "freq": f, "K": K,
                                       "sharpe": sharpe(ex), "sharpe_sem": sharpe(ex[fora]),
                                       "vol": ret.std() * np.sqrt(252), "vol_sem": ret[fora].std() * np.sqrt(252)})
    d = pd.DataFrame(linhas)
    DEST.mkdir(parents=True, exist_ok=True)
    d.to_csv(DEST / "sensibilidade_mmxm3.csv", index=False)

    out = ["# Sensibilidade ao episódio MMXM3 (D38)", "",
           f"Métricas com e sem os pregões de {INI} a {FIM}. Sharpe = excesso diário médio sobre o CDI / "
           "desvio × √252 (não é o `sharpe_cdi` oficial; serve para comparar com/sem). Média nos 7 K e 5 "
           "frequências, custo 0.", "",
           "| Configuração | Sharpe | Sharpe sem episódio | Vol. | Vol. sem episódio |", "|---|---|---|---|---|"]
    for (r, m, e), g in d.groupby(["regra", "metodo", "medida"]):
        out.append(f"| {REGRAS[r]} · {MET[m]} · {MED[e]} | {g.sharpe.mean():.2f} | {g.sharpe_sem.mean():.2f} | "
                   f"{g.vol.mean():.1%} | {g.vol_sem.mean():.1%} |")
    out += ["", "## Efeito pareado CVaR − variância no Sharpe (mesmo K e frequência)", "",
            "| Cenário · método | Com episódio | Sem episódio |", "|---|---|---|"]
    for r in REGRAS:
        for m in MET:
            a = d[(d.regra == r) & (d.metodo == m) & (d.medida == "cvar")].set_index(["K", "freq"])
            b = d[(d.regra == r) & (d.metodo == m) & (d.medida == "variancia")].set_index(["K", "freq"])
            out.append(f"| {REGRAS[r]} · {MET[m]} | {(a.sharpe - b.sharpe).mean():+.3f} | "
                       f"{(a.sharpe_sem - b.sharpe_sem).mean():+.3f} |")
    out += ["", "## Efeito pareado CPLEX − NSGA-II no Sharpe", "",
            "| Cenário · medida | Com episódio | Sem episódio |", "|---|---|---|"]
    for r in REGRAS:
        for e in MED:
            a = d[(d.regra == r) & (d.metodo == "cplex") & (d.medida == e)].set_index(["K", "freq"])
            b = d[(d.regra == r) & (d.metodo == "nsga2") & (d.medida == e)].set_index(["K", "freq"])
            out.append(f"| {REGRAS[r]} · {MED[e]} | {(a.sharpe - b.sharpe).mean():+.3f} | "
                       f"{(a.sharpe_sem - b.sharpe_sem).mean():+.3f} |")
    pes = []
    for r in REGRAS:
        for m in MET:
            for e in MED:
                p = pd.read_csv(GRADE / f"{r}_{e}_{m}" / "pesos_decisoes.csv")
                x = p[p.ticker == "MMXM3"]
                pes.append(f"| {REGRAS[r]} · {MET[m]} · {MED[e]} | {x.t.nunique()} | "
                           f"{x.peso.mean() if len(x) else 0:.1%} | {x.t.min() if len(x) else '—'} | "
                           f"{x.t.max() if len(x) else '—'} |")
    out += ["", "## Presença da MMXM3 nas decisões", "",
            "| Configuração | Datas com MMXM3 | Peso médio | Primeira | Última |", "|---|---|---|---|---|"] + pes
    (DEST / "sensibilidade_mmxm3.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
