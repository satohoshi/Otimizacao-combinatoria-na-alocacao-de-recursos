"""compara as 4 configs do backtest (sigma amostral/lw x max sharpe/var min)
le results/backtest/<sigma>_<regra>_<b3|semb3>/metricas.csv

uso:
    python3 scripts/comparar_backtests.py             # yahoo + b3
    python3 scripts/comparar_backtests.py --sem-b3    # so yahoo
saidas em results/backtest/:
    comparacao_<b3|semb3>.csv, comparacao_<b3|semb3>.png/.pdf
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest  # noqa: E402

CONFIGS = {("amostral", "max_sharpe"): "Σ amostral, máx. Sharpe",
           ("lw", "max_sharpe"): "Σ Ledoit-Wolf, máx. Sharpe",
           ("amostral", "var_min"): "Σ amostral, var. mínima",
           ("lw", "var_min"): "Σ Ledoit-Wolf, var. mínima"}
COLS = ["retorno_anual", "volatilidade_anual", "sharpe_cdi", "drawdown_max", "turnover_anual"]


def carregar(base):
    partes, bench = [], None
    for (sigma, regra), nome in CONFIGS.items():
        m = pd.read_csv(backtest.SAIDA / f"{sigma}_{regra}_{base}" / "metricas.csv")
        bench = m[m.estrategia.isin(["Ibovespa", "CDI"])]
        m = m[m.tratamento == "degrau_neutralizado"].assign(sigma=sigma, regra=regra, config=nome)
        partes.append(m)
    return pd.concat(partes, ignore_index=True), bench


def grafico(comp, bench, destino, arquivo="comparacao"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cores = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    estilos = {"max_sharpe": "-", "var_min": "--"}
    marcas = {"amostral": "o", "lw": "s"}
    tinta, tinta2, muda, grade = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": tinta2,
                         "xtick.color": muda, "ytick.color": muda})
    ks = sorted(backtest.CARTEIRAS)
    d = comp[(comp.freq == "anual") & (comp.custo == 0)]
    painéis = [("sharpe_cdi", "Sharpe realizado contra o CDI", 1),
               ("volatilidade_anual", "Volatilidade anualizada (%)", 100),
               ("drawdown_max", "Drawdown máximo (%)", 100)]
    fig, eixos = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, (col, titulo, esc) in zip(eixos, painéis):
        for cor, ((sigma, regra), nome) in zip(cores, CONFIGS.items()):
            x = d[(d.sigma == sigma) & (d.regra == regra)].set_index("K").loc[ks]
            ax.plot(range(len(ks)), x[col] * esc, color=cor, ls=estilos[regra], marker=marcas[sigma],
                    ms=5, lw=1.4, label=nome)
        ib = bench.loc[bench.estrategia == "Ibovespa", col].iloc[0] * esc
        ax.axhline(ib, color=tinta2, lw=1)
        ax.annotate("Ibovespa", (len(ks) - 1, ib), xytext=(0, 4), textcoords="offset points",
                    ha="right", fontsize=8, color=tinta2)
        ax.set_xticks(range(len(ks)), [str(k) for k in ks])
        ax.set_xlabel("K (nº de ativos)")
        ax.set_title(titulo, fontsize=10, color=tinta, loc="left")
        ax.grid(color=grade, lw=0.6)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    h, r = eixos[0].get_legend_handles_labels()
    fig.legend(h, r, frameon=False, fontsize=8.5, ncol=4, loc="lower center")
    fig.suptitle("Rebalanceamento anual, sem custos, 2014–2026", fontsize=10, color=tinta, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(destino / f"{arquivo}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-b3", action="store_true", help="só Yahoo (sem as séries do COTAHIST)")
    base = "semb3" if ap.parse_args().sem_b3 else "b3"
    comp, bench = carregar(base)
    tab = comp[["config", "sigma", "regra", "K", "freq", "custo"] + COLS]
    tab.to_csv(backtest.SAIDA / f"comparacao_{base}.csv", index=False)
    grafico(comp, bench, backtest.SAIDA, f"comparacao_{base}")

    pd.set_option("display.width", 220)
    for freq in ["anual", "trimestral"]:
        x = comp[(comp.freq == freq) & (comp.custo == 0)]
        print(f"\n== {freq}, sem custo")
        for col in ["retorno_anual", "volatilidade_anual", "sharpe_cdi", "drawdown_max", "turnover_anual"]:
            print(f"-- {col}")
            print(x.pivot_table(index="config", columns="K", values=col).loc[list(CONFIGS.values())]
                  .to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n" + bench[["estrategia"] + COLS[:4]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
