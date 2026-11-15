"""testes de significancia das diferencas pareadas

pra cada comparacao os pares sao duas configs q so mudam no fator testado, com
mesmo K e frequencia (igual no efeitos_pareados)

1. wilcoxon pareado (bicaudal) nas diferencas de sharpe_cdi. os pares n sao
   independentes (K vizinhos e frequencias diferentes dividem ativos e periodo),
   entao o p-valor tende a ser otimista, serve so de primeira olhada
2. bootstrap em blocos moveis (blocos de 21 pregoes, 2000 reamostragens) dos
   retornos diarios em excesso ao cdi. usa a mesma reamostragem de dias pra todas
   as series, assim mantem a dependencia entre as carteiras. estatistica: media da
   diferenca de sharpe diario anualizado (media/desvio * raiz(252)). sai o ic de
   95% (percentis) e o p-valor bicaudal

saida: results/grade/comparacao/testes_significancia.csv e .md
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RAIZ = Path(__file__).resolve().parents[1]
GRADE = RAIZ / "results" / "grade"
DEST = GRADE / "comparacao"
FREQS = ["sem_rebal", "anual", "semestral", "trimestral", "mensal"]
KS = [3, 4, 5, 9, 15, 20, 30]
REGRAS = {"max_sharpe": "Máximo Sharpe", "var_min": "Risco mínimo"}
MET = {"cplex": "CPLEX", "nsga2": "NSGA-II"}
MED = {"variancia": "variância", "cvar": "CVaR"}
BLOCO, B, SEED = 21, 2000, 0


def carregar():
    ex, sh = {}, {}
    for r in REGRAS:
        for m in MET:
            for e in MED:
                p = GRADE / f"{r}_{e}_{m}"
                v = pd.read_csv(p / "valor_diario.csv", index_col="data", parse_dates=True)
                cdi = v["CDI"].pct_change()
                met = pd.read_csv(p / "metricas.csv")
                met = met[(met.tratamento == "degrau_neutralizado") & (met.custo == 0)]
                for f in FREQS:
                    for K in KS:
                        col = f"K{K}_{f}_c0.00%"
                        ex[(r, m, e, f, K)] = (v[col].pct_change() - cdi).iloc[1:].values
                        x = met[(met.freq == f) & (met.K == K)].sharpe_cdi
                        sh[(r, m, e, f, K)] = float(x.iloc[0])
    return ex, sh


def comparacoes():
    """(fator, rotulo do primeiro - segundo, o q fica fixo, lista de pares (a, b))"""
    out = []
    for r in REGRAS:
        for e in MED:
            out.append(("método", "CPLEX − NSGA-II", f"{REGRAS[r]} · {MED[e]}",
                        [((r, "cplex", e, f, K), (r, "nsga2", e, f, K)) for f in FREQS for K in KS]))
    for r in REGRAS:
        for m in MET:
            out.append(("medida", "CVaR − variância", f"{REGRAS[r]} · {MET[m]}",
                        [((r, m, "cvar", f, K), (r, m, "variancia", f, K)) for f in FREQS for K in KS]))
    for m in MET:
        for e in MED:
            out.append(("cenário", "Máx. Sharpe − risco mín.", f"{MET[m]} · {MED[e]}",
                        [(("max_sharpe", m, e, f, K), ("var_min", m, e, f, K)) for f in FREQS for K in KS]))
    for r in REGRAS:
        for a, b in [("trimestral", "anual"), ("mensal", "trimestral"), ("anual", "sem_rebal")]:
            out.append(("frequência", f"{a} − {b}".replace("sem_rebal", "sem rebal."), REGRAS[r],
                        [((r, m, e, a, K), (r, m, e, b, K)) for m in MET for e in MED for K in KS]))
    return out


def main():
    ex, sh = carregar()
    n = len(next(iter(ex.values())))
    rng = np.random.default_rng(SEED)
    nb = int(np.ceil(n / BLOCO))
    inicios = rng.integers(0, n - BLOCO + 1, size=(B, nb))
    idx = (inicios[:, :, None] + np.arange(BLOCO)).reshape(B, -1)[:, :n]       # B x n

    cache = {}

    def sharpe_boot(k):
        if k not in cache:
            x = ex[k][idx]
            cache[k] = x.mean(axis=1) / x.std(axis=1, ddof=1) * np.sqrt(252)
        return cache[k]

    def sharpe_pont(k):
        x = ex[k]
        return x.mean() / x.std(ddof=1) * np.sqrt(252)

    linhas = []
    for fator, rotulo, fixo, pares in comparacoes():
        dif = np.array([sh[a] - sh[b] for a, b in pares])
        w = wilcoxon(dif) if np.any(dif != 0) else None
        pont = np.mean([sharpe_pont(a) - sharpe_pont(b) for a, b in pares])
        boot = np.mean([sharpe_boot(a) - sharpe_boot(b) for a, b in pares], axis=0)
        p_boot = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
        linhas.append({"fator": fator, "comparacao": rotulo, "fixo": fixo, "pares": len(pares),
                       "dif_media_oficial": dif.mean(), "dif_mediana_oficial": np.median(dif),
                       "pct_positivos": (dif > 0).mean(), "wilcoxon_p": w.pvalue if w else np.nan,
                       "dif_media_diaria": pont, "ic95_inf": np.percentile(boot, 2.5),
                       "ic95_sup": np.percentile(boot, 97.5), "bootstrap_p": p_boot})
        print(f"{fator:10s} {rotulo:26s} {fixo:28s} média {dif.mean():+.3f}  Wilcoxon p={linhas[-1]['wilcoxon_p']:.3f}"
              f"  boot IC [{linhas[-1]['ic95_inf']:+.3f}, {linhas[-1]['ic95_sup']:+.3f}] p={p_boot:.3f}", flush=True)
    d = pd.DataFrame(linhas)
    DEST.mkdir(parents=True, exist_ok=True)
    d.to_csv(DEST / "testes_significancia.csv", index=False)


if __name__ == "__main__":
    main()
