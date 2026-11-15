"""compara no backtest 2014-2026 as 4 combinacoes {cplex, nsga2} x {variancia, cvar}
mesma regra (max sharpe ex-ante), mesmos K, datas, custos e dados, so muda a medida e o metodo

le results/backtest/<sigma>_max_sharpe_<b3|semb3>[_<medida>_<metodo>]/
(decisoes.csv, pesos_decisoes.csv, metricas.csv, valor_diario.csv)

uso:
    python3 scripts/comparar_combinacoes.py [--sigma amostral] [--sem-b3] [--freq anual]
saidas em results/backtest/combinacoes_<sigma>_<b3|semb3>/:
    metricas.csv        retorno, vol, sharpe, drawdown, turnover por combinacao, K, frequencia e custo
    tempo.csv           tempo de otimizacao total e por decisao
    sobreposicao.csv    sobreposicao media de ativos entre combinacoes (jaccard) por K
    sharpe_vol_dd, valor_acumulado, tempo_otimizacao, ativos_K<k> (.png/.pdf)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest  # noqa: E402

COMBOS = {("variancia", "cplex"): "CPLEX · variância", ("cvar", "cplex"): "CPLEX · CVaR",
          ("variancia", "nsga2"): "NSGA-II · variância", ("cvar", "nsga2"): "NSGA-II · CVaR"}
COR = {("variancia", "cplex"): "#2a78d6", ("cvar", "cplex"): "#1baf7a",
       ("variancia", "nsga2"): "#eb6834", ("cvar", "nsga2"): "#eda100"}
TINTA, TINTA2, MUDA, GRADE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def pasta(sigma, base, medida, metodo):
    sufixo = "" if (medida, metodo) == ("variancia", "cplex") else f"_{medida}_{metodo}"
    return backtest.SAIDA / f"{sigma}_max_sharpe_{base}{sufixo}"


def estilo(ax):
    ax.grid(color=GRADE, lw=0.6)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def salvar(fig, destino, nome):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(destino / f"{nome}.{ext}", dpi=200, facecolor="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", default="amostral")
    ap.add_argument("--sem-b3", action="store_true")
    ap.add_argument("--freq", default="anual", help="frequência dos gráficos (precisa existir nas 4 combinações)")
    args = ap.parse_args()
    base = "semb3" if args.sem_b3 else "b3"
    destino = backtest.SAIDA / f"combinacoes_{args.sigma}_{base}"
    destino.mkdir(parents=True, exist_ok=True)

    met, dec, pes, val = [], [], [], {}
    for (medida, metodo), nome in COMBOS.items():
        d = pasta(args.sigma, base, medida, metodo)
        if not (d / "metricas.csv").exists():
            print(f"ausente: {d.relative_to(backtest.RAIZ)} (rode backtest.py --medida {medida} --metodo {metodo})")
            continue
        m = pd.read_csv(d / "metricas.csv")
        met.append(m[m.tratamento == "degrau_neutralizado"].assign(combo=nome, medida=medida, metodo=metodo))
        dec.append(pd.read_csv(d / "decisoes.csv", parse_dates=["t"]).assign(combo=nome, medida=medida, metodo=metodo))
        pes.append(pd.read_csv(d / "pesos_decisoes.csv", parse_dates=["t"]).assign(combo=nome))
        val[nome] = pd.read_csv(d / "valor_diario.csv", index_col="data", parse_dates=True)
        bench = m[m.estrategia.isin(["Ibovespa", "CDI"])]
    if len(met) < 2:
        sys.exit("faltam combinações")
    met, dec, pes = pd.concat(met), pd.concat(dec), pd.concat(pes)
    nomes = [n for n in COMBOS.values() if n in set(met.combo)]
    chave = {n: k for k, n in COMBOS.items()}
    ks = sorted(met.K.dropna().unique().astype(int))
    cols = ["retorno_anual", "volatilidade_anual", "sharpe_cdi", "drawdown_max", "turnover_anual"]
    met[["combo", "K", "freq", "custo"] + cols].to_csv(destino / "metricas.csv", index=False)

    # tempo de otimizacao: soma das decisoes de cada (combinacao, K) nas datas da frequencia
    tempo = (dec.groupby(["combo", "K"]).agg(decisoes=("t", "nunique"), tempo_total_s=("tempo_total_s", "sum"),
                                              tempo_por_decisao_s=("tempo_total_s", "mean"),
                                              resolucoes_por_decisao=("n_resolucoes", "mean")).reset_index())
    tempo.to_csv(destino / "tempo.csv", index=False)

    # sobreposicao de ativos (jaccard) entre combinacoes nas datas em comum, por K
    linhas = []
    for K in ks:
        sets = {n: pes[(pes.combo == n) & (pes.K == K)].groupby("t").ticker.agg(frozenset) for n in nomes}
        for i, a in enumerate(nomes):
            for b in nomes[i + 1:]:
                comum = sets[a].index.intersection(sets[b].index)
                j = [len(sets[a][t] & sets[b][t]) / len(sets[a][t] | sets[b][t]) for t in comum]
                linhas.append({"K": K, "a": a, "b": b, "datas": len(comum), "jaccard_medio": np.mean(j)})
    pd.DataFrame(linhas).to_csv(destino / "sobreposicao.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TINTA2,
                         "xtick.color": MUDA, "ytick.color": MUDA})
    marca = {"cplex": "o", "nsga2": "s"}
    est = {"variancia": "-", "cvar": "--"}
    d0 = met[(met.freq == args.freq) & (met.custo == 0)]

    # fig 1: sharpe, vol e drawdown por K
    fig, eixos = plt.subplots(1, 3, figsize=(12.5, 4))
    for ax, (col, tit, esc) in zip(eixos, [("sharpe_cdi", "Sharpe realizado contra o CDI", 1),
                                            ("volatilidade_anual", "Volatilidade anualizada (%)", 100),
                                            ("drawdown_max", "Drawdown máximo (%)", 100)]):
        for n in nomes:
            medida, metodo = chave[n]
            x = d0[d0.combo == n].set_index("K").reindex(ks)
            ax.plot(range(len(ks)), x[col] * esc, color=COR[(medida, metodo)], ls=est[medida],
                    marker=marca[metodo], ms=5, lw=1.5, label=n)
        ib = bench.loc[bench.estrategia == "Ibovespa", col].iloc[0] * esc
        ax.axhline(ib, color=TINTA2, lw=1)
        ax.annotate("Ibovespa", (len(ks) - 1, ib), xytext=(0, 4), textcoords="offset points", ha="right",
                    fontsize=8, color=TINTA2)
        ax.set_xticks(range(len(ks)), [str(k) for k in ks])
        ax.set_xlabel("K (nº de ativos)")
        ax.set_title(tit, fontsize=10, color=TINTA, loc="left")
        estilo(ax)
    h, r = eixos[0].get_legend_handles_labels()
    fig.legend(h, r, frameon=False, fontsize=8.5, ncol=4, loc="lower center")
    fig.suptitle(f"Máximo Sharpe, rebalanceamento {args.freq}, sem custos — {args.sigma}", fontsize=10,
                 color=TINTA, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(destino / f"sharpe_vol_dd.{ext}", dpi=200, facecolor="white")
    plt.close(fig)

    # fig 2: valor acumulado (K = 9) e drawdown
    k_fig = 9 if 9 in ks else ks[len(ks) // 2]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for n in nomes:
        medida, metodo = chave[n]
        s = val[n][f"K{k_fig}_{args.freq}_c0.00%"]
        ax.plot(s.index, s / 1000, color=COR[(medida, metodo)], ls=est[medida], lw=1.5, label=n)
        ax2.plot(s.index, (s / s.cummax() - 1) * 100, color=COR[(medida, metodo)], ls=est[medida], lw=1.1)
    ib = val[nomes[0]]["Ibovespa"]
    ax.plot(ib.index, ib / 1000, color=TINTA2, lw=1.3, label="Ibovespa")
    ax2.plot(ib.index, (ib / ib.cummax() - 1) * 100, color=TINTA2, lw=1)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0f}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("Valor (R$ mil, escala log)")
    ax2.set_ylabel("Drawdown (%)")
    ax.set_title(f"(a) Valor da carteira — K = {k_fig}, {args.freq}, sem custos", fontsize=10, color=TINTA, loc="left")
    ax2.set_title("(b) Queda em relação ao pico anterior", fontsize=10, color=TINTA, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    estilo(ax)
    estilo(ax2)
    salvar(fig, destino, "valor_acumulado")
    plt.close(fig)

    # fig 3: tempo de otimizacao por decisao
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    larg = 0.8 / len(nomes)
    for i, n in enumerate(nomes):
        x = tempo[tempo.combo == n].set_index("K").reindex(ks)
        ax.bar(np.arange(len(ks)) + (i - (len(nomes) - 1) / 2) * larg, x.tempo_por_decisao_s, larg * 0.92,
               color=COR[chave[n]], label=n)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ks)), [str(k) for k in ks])
    ax.set_xlabel("K (nº de ativos)")
    ax.set_ylabel("Tempo por decisão (s, escala log)")
    ax.set_title("(a) Tempo de otimização por data de decisão", fontsize=10, color=TINTA, loc="left")
    ax.legend(frameon=False, fontsize=8)
    estilo(ax)
    # (b) sharpe realizado x tempo, quanto ganha pelo tempo gasto (K = k_fig)
    for n in nomes:
        t = tempo[(tempo.combo == n)].tempo_por_decisao_s.mean()
        sr = d0[(d0.combo == n) & (d0.K == k_fig)].sharpe_cdi.iloc[0]
        ax2.scatter(t, sr, s=70, color=COR[chave[n]], zorder=3)
        ax2.annotate(n, (t, sr), xytext=(6, 4), textcoords="offset points", fontsize=8, color=TINTA2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Tempo médio por decisão (s, média dos K, escala log)")
    ax2.set_ylabel(f"Sharpe realizado (K = {k_fig})")
    ax2.set_title("(b) Qualidade realizada × custo computacional", fontsize=10, color=TINTA, loc="left")
    estilo(ax2)
    salvar(fig, destino, "tempo_otimizacao")
    plt.close(fig)

    # fig 4 (por K): ativos mais frequentes nas decisoes, por combinacao
    for K in ks:
        p = pes[pes.K == K]
        freq = pd.DataFrame({n: p[p.combo == n].groupby("ticker").t.nunique() / p[p.combo == n].t.nunique()
                             for n in nomes}).fillna(0)
        top = freq.mean(axis=1).sort_values(ascending=False).index[:25]
        v = freq.loc[top, nomes].values * 100
        fig, ax = plt.subplots(figsize=(6.2, 0.27 * len(top) + 1.6))
        ax.imshow(v, aspect="auto", cmap="Blues", vmin=0, vmax=max(v.max(), 1))
        for (a, b), x in np.ndenumerate(v):
            if x >= 0.5:
                ax.text(b, a, f"{x:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if x > 0.6 * v.max() else TINTA)
        ax.set_xticks(range(len(nomes)), [n.replace(" · ", "\n") for n in nomes], fontsize=8)
        ax.set_yticks(range(len(top)), top, fontsize=8)
        ax.tick_params(length=0)
        for lado in ax.spines.values():
            lado.set_visible(False)
        ax.set_title(f"Presença do ativo nas decisões (%) — K = {K}", fontsize=10, color=TINTA, loc="left")
        salvar(fig, destino, f"ativos_K{K}")
        plt.close(fig)

    pd.set_option("display.width", 220)
    for col in ["sharpe_cdi", "retorno_anual", "volatilidade_anual", "drawdown_max"]:
        print(f"-- {col} ({args.freq}, sem custo)")
        print(d0.pivot_table(index="combo", columns="K", values=col).reindex(nomes).to_string(
            float_format=lambda x: f"{x:.3f}"))
    print("-- tempo por decisão (s)")
    print(tempo.pivot_table(index="combo", columns="K", values="tempo_por_decisao_s").reindex(nomes).to_string(
        float_format=lambda x: f"{x:.2f}"))
    print(f"Saídas em {destino.relative_to(backtest.RAIZ)}")


if __name__ == "__main__":
    main()
