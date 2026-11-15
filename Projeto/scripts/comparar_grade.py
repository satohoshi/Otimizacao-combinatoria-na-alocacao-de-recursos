"""grade completa do backtest 2014-2026:
{max sharpe, risco min} x {cplex, nsga2} x {variancia, cvar} x
{sem rebal, anual, semestral, trimestral, mensal}, sigma amostral, yahoo + b3

le results/grade/<regra>_<medida>_<metodo>/ (do results/grade/rodar_grade.sh) e grava:
    results/grade/comparacao/   tabelas (csv/md) e figuras
    ../graficos_resultados/     copias organizadas pelas secoes do texto

uso:  ../venv/bin/python3 scripts/comparar_grade.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest  # noqa: E402

RAIZ = backtest.RAIZ
GRADE = RAIZ / "results" / "grade"
DEST = GRADE / "comparacao"
FOTO = RAIZ.parent / "graficos_resultados"

REGRAS = {"max_sharpe": "Máximo Sharpe", "var_min": "Risco mínimo"}
COMBOS = {("cplex", "variancia"): "CPLEX · variância", ("cplex", "cvar"): "CPLEX · CVaR",
          ("nsga2", "variancia"): "NSGA-II · variância", ("nsga2", "cvar"): "NSGA-II · CVaR"}
COR = {("cplex", "variancia"): "#2a78d6", ("cplex", "cvar"): "#1baf7a",
       ("nsga2", "variancia"): "#eb6834", ("nsga2", "cvar"): "#eda100"}
ESTILO = {"variancia": "-", "cvar": "--"}
MARCA = {"cplex": "o", "nsga2": "s"}
MARCA_REGRA = {"max_sharpe": "o", "var_min": "^"}
FREQS = ["sem_rebal", "anual", "semestral", "trimestral", "mensal"]
ROT_FREQ = {"sem_rebal": "sem rebal.", "anual": "anual", "semestral": "semestral",
            "trimestral": "trimestral", "mensal": "mensal"}
KS = [3, 4, 5, 9, 15, 20, 30]
K_FIG = 9
CUSTO = 0.0          # figuras sem custo, as tabelas tem os dois
TINTA, TINTA2, MUDA, GRADE_COR = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
METRICAS = [("sharpe_cdi", "Sharpe realizado contra o CDI", 1),
            ("volatilidade_anual", "Volatilidade anualizada (%)", 100),
            ("drawdown_max", "Drawdown máximo (%)", 100)]


def cfg(regra, metodo, medida):
    return f"{regra}_{medida}_{metodo}"


def carregar():
    met, dec, pes, val = [], [], [], {}
    bench = None
    for regra in REGRAS:
        for (metodo, medida) in COMBOS:
            d = GRADE / cfg(regra, metodo, medida)
            if not (d / "metricas.csv").exists():
                print(f"ausente: {d.name}")
                continue
            tag = dict(regra=regra, metodo=metodo, medida=medida)
            m = pd.read_csv(d / "metricas.csv")
            bench = m[m.estrategia.isin(["Ibovespa", "CDI"])].set_index("estrategia")
            met.append(m[m.tratamento == "degrau_neutralizado"].assign(**tag))
            dec.append(pd.read_csv(d / "decisoes.csv", parse_dates=["t"]).assign(**tag))
            pes.append(pd.read_csv(d / "pesos_decisoes.csv", parse_dates=["t"]).assign(**tag))
            val[(regra, metodo, medida)] = pd.read_csv(d / "valor_diario.csv", index_col="data", parse_dates=True)
    met = pd.concat(met)
    met = met[met.freq.isin(FREQS)].copy()
    met["K"] = met.K.astype(int)
    dec, pes = pd.concat(dec), pd.concat(pes)
    dec["K"], pes["K"] = dec.K.astype(int), pes.K.astype(int)
    # so as datas mensais (o nsga2 com max sharpe tem as semanais tb)
    cal = pd.DatetimeIndex(next(iter(val.values())).index)
    mensais = set(backtest.datas_frequencia(cal, "mensal"))
    dec, pes = dec[dec.t.isin(mensais)], pes[pes.t.isin(mensais)]
    return met, dec, pes, val, bench


# graficos
def preparar_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TINTA2,
                         "xtick.color": MUDA, "ytick.color": MUDA})
    return matplotlib, plt


def estilo(ax):
    ax.grid(color=GRADE_COR, lw=0.6)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def salvar(fig, pasta, nome):
    pasta.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(pasta / f"{nome}.{ext}", dpi=200, facecolor="white")


def titulo(fig, texto):
    fig.suptitle(texto, fontsize=10, color=TINTA, x=0.01, ha="left")


def fig_mapa_sharpe(plt, met, bench, pasta):
    """sharpe medio nos 7 K pra cada (regra, combinacao, frequencia)"""
    d = met[met.custo == CUSTO]
    tabs = {r: d[d.regra == r].pivot_table(index=["metodo", "medida"], columns="freq", values="sharpe_cdi",
                                           aggfunc="mean").reindex(index=list(COMBOS), columns=FREQS)
            for r in REGRAS}
    vmin = min(t.min().min() for t in tabs.values())
    vmax = max(t.max().max() for t in tabs.values())
    ib = bench.at["Ibovespa", "sharpe_cdi"]
    fig, eixos = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, (r, t) in zip(eixos, tabs.items()):
        v = t.values
        ax.imshow(v, cmap="Blues", vmin=min(vmin, 0), vmax=vmax, aspect="auto")
        for (a, b), x in np.ndenumerate(v):
            if np.isnan(x):
                continue
            ax.text(b, a, f"{x:.2f}".replace(".", ","), ha="center", va="center", fontsize=9,
                    color="white" if x > 0.6 * vmax else TINTA, fontweight="bold" if x < ib else "normal")
        ax.set_xticks(range(len(FREQS)), [ROT_FREQ[f] for f in FREQS], fontsize=8.5)
        ax.set_yticks(range(len(COMBOS)), list(COMBOS.values()), fontsize=8.5)
        ax.tick_params(length=0)
        for lado in ax.spines.values():
            lado.set_visible(False)
        ax.set_title(f"({'ab'[list(REGRAS).index(r)]}) {REGRAS[r]}", fontsize=10, color=TINTA, loc="left")
    titulo(fig, f"Sharpe realizado médio nos 7 valores de K — 2014-2026, sem custos "
                f"(Ibovespa = {ib:.2f}; em negrito, abaixo do Ibovespa)".replace(".", ",", 1))
    salvar(fig, pasta, "mapa_sharpe")
    plt.close(fig)


def fig_risco_retorno(plt, met, bench, pasta):
    """retorno x vol realizados (media nos K) de cada config"""
    d = met[met.custo == CUSTO].groupby(["regra", "metodo", "medida", "freq"])[
        ["retorno_anual", "volatilidade_anual", "sharpe_cdi"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in REGRAS:
        for (metodo, medida), nome in COMBOS.items():
            x = d[(d.regra == r) & (d.metodo == metodo) & (d.medida == medida)].set_index("freq").reindex(FREQS)
            ax.plot(x.volatilidade_anual * 100, x.retorno_anual * 100, color=COR[(metodo, medida)],
                    ls=ESTILO[medida], lw=0.9, alpha=0.6)
            ax.scatter(x.volatilidade_anual * 100, x.retorno_anual * 100, color=COR[(metodo, medida)],
                       marker=MARCA_REGRA[r], s=[18, 26, 34, 42, 54], zorder=3,
                       label=f"{REGRAS[r]} · {nome}")
            fim = x.loc["mensal"]
            ax.annotate("mensal", (fim.volatilidade_anual * 100, fim.retorno_anual * 100), xytext=(4, 2),
                        textcoords="offset points", fontsize=6.5, color=MUDA)
    for nome in ("Ibovespa", "CDI"):
        b = bench.loc[nome]
        ax.scatter(b.volatilidade_anual * 100, b.retorno_anual * 100, color=TINTA, s=30, zorder=3)
        ax.annotate(nome, (b.volatilidade_anual * 100, b.retorno_anual * 100), xytext=(6, -3),
                    textcoords="offset points", fontsize=8.5, color=TINTA)
    ax.set_xlabel("Volatilidade anualizada realizada (%)")
    ax.set_ylabel("Retorno anualizado realizado (%)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    estilo(ax)
    titulo(fig, "Retorno × risco realizados, média nos 7 K, sem custos\n"
                "● máximo Sharpe, ▲ risco mínimo; o ponto cresce de 'sem rebal.' a 'mensal'")
    salvar(fig, pasta, "risco_retorno")
    plt.close(fig)


def fig_por_frequencia(plt, met, bench, regra, pasta):
    """metricas (media nos K) por frequencia, uma linha por combinacao"""
    d = met[(met.custo == CUSTO) & (met.regra == regra)]
    fig, eixos = plt.subplots(1, 3, figsize=(12.5, 4))
    for ax, (col, tit, esc) in zip(eixos, METRICAS):
        for (metodo, medida), nome in COMBOS.items():
            x = d[(d.metodo == metodo) & (d.medida == medida)].groupby("freq")[col].mean().reindex(FREQS)
            ax.plot(range(len(FREQS)), x * esc, color=COR[(metodo, medida)], ls=ESTILO[medida],
                    marker=MARCA[metodo], ms=5, lw=1.5, label=nome)
        ib = bench.at["Ibovespa", col] * esc
        ax.axhline(ib, color=TINTA2, lw=1)
        ax.annotate("Ibovespa", (len(FREQS) - 1, ib), xytext=(0, 4), textcoords="offset points", ha="right",
                    fontsize=8, color=TINTA2)
        ax.set_xticks(range(len(FREQS)), [ROT_FREQ[f] for f in FREQS], fontsize=8.5)
        ax.set_title(tit, fontsize=10, color=TINTA, loc="left")
        estilo(ax)
    h, r = eixos[0].get_legend_handles_labels()
    fig.legend(h, r, frameon=False, fontsize=8.5, ncol=4, loc="lower center")
    titulo(fig, f"{REGRAS[regra]} — média nos 7 K por frequência de rebalanceamento, sem custos")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    pasta.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(pasta / f"metricas_por_frequencia.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def fig_por_k(plt, met, bench, regra, freq, pasta):
    """sharpe, vol e drawdown por K pra cada cenario e frequencia"""
    d = met[(met.custo == CUSTO) & (met.regra == regra) & (met.freq == freq)]
    fig, eixos = plt.subplots(1, 3, figsize=(12.5, 4))
    for ax, (col, tit, esc) in zip(eixos, METRICAS):
        for (metodo, medida), nome in COMBOS.items():
            x = d[(d.metodo == metodo) & (d.medida == medida)].set_index("K").reindex(KS)
            ax.plot(range(len(KS)), x[col] * esc, color=COR[(metodo, medida)], ls=ESTILO[medida],
                    marker=MARCA[metodo], ms=5, lw=1.5, label=nome)
        ib = bench.at["Ibovespa", col] * esc
        ax.axhline(ib, color=TINTA2, lw=1)
        ax.annotate("Ibovespa", (len(KS) - 1, ib), xytext=(0, 4), textcoords="offset points", ha="right",
                    fontsize=8, color=TINTA2)
        ax.set_xticks(range(len(KS)), [str(k) for k in KS])
        ax.set_xlabel("K (nº de ativos)")
        ax.set_title(tit, fontsize=10, color=TINTA, loc="left")
        estilo(ax)
    h, r = eixos[0].get_legend_handles_labels()
    fig.legend(h, r, frameon=False, fontsize=8.5, ncol=4, loc="lower center")
    titulo(fig, f"{REGRAS[regra]}, rebalanceamento {ROT_FREQ[freq]}, sem custos")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    pasta.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(pasta / f"metricas_por_k_{freq}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def fig_valor(matplotlib, plt, val, regra, freq, pasta):
    """valor acumulado (K = 9) das 4 combinacoes e drawdown"""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for (metodo, medida), nome in COMBOS.items():
        v = val.get((regra, metodo, medida))
        col = f"K{K_FIG}_{freq}_c{CUSTO:.2%}"
        if v is None or col not in v:
            continue
        s = v[col]
        ax.plot(s.index, s / 1000, color=COR[(metodo, medida)], ls=ESTILO[medida], lw=1.5, label=nome)
        ax2.plot(s.index, (s / s.cummax() - 1) * 100, color=COR[(metodo, medida)], ls=ESTILO[medida], lw=1.1)
    v = next(iter(val.values()))
    for nome, ls in [("Ibovespa", "-"), ("CDI", ":")]:
        s = v[nome]
        ax.plot(s.index, s / 1000, color=TINTA2, lw=1.3, ls=ls, label=nome)
        ax2.plot(s.index, (s / s.cummax() - 1) * 100, color=TINTA2, lw=1, ls=ls)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0f}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("Valor (R$ mil, escala log)")
    ax2.set_ylabel("Drawdown (%)")
    ax.set_title(f"(a) Valor da carteira — {REGRAS[regra]}, K = {K_FIG}, {ROT_FREQ[freq]}, sem custos",
                 fontsize=10, color=TINTA, loc="left")
    ax2.set_title("(b) Queda em relação ao pico anterior", fontsize=10, color=TINTA, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    estilo(ax)
    estilo(ax2)
    salvar(fig, pasta, f"valor_acumulado_{freq}")
    plt.close(fig)


COR_FREQ = dict(zip(FREQS, ["#52514e", "#2a78d6", "#1baf7a", "#eda100", "#eb6834"]))
ROT_FREQ_LONGO = {"sem_rebal": "sem rebalanceamento", "anual": "anual", "semestral": "semestral",
                  "trimestral": "trimestral", "mensal": "mensal"}


MARCOS = [2014, 2016, 2018, 2020, 2022, 2024, 2026]   # marcos do eixo x
SERIES_2X2 = FREQS + ["Ibovespa", "CDI"]
ROT_SERIE = {**ROT_FREQ_LONGO, "Ibovespa": "Ibovespa", "CDI": "CDI"}
ROT_CURTO = {**ROT_FREQ, "Ibovespa": "Ibovespa", "CDI": "CDI"}


def janelas_bienais():
    return [f"{a}–{b}" for a, b in zip(MARCOS, MARCOS[1:])]


def cagr_bienal(s):
    """cagr (% a.a.) entre os marcos (primeiro pregao do ano) e no periodo todo
    cagr = (V_fim / V_ini)^(1/anos) - 1, anos = dias corridos / 365.25"""
    s = s.dropna()
    pts = [s.index[s.index.searchsorted(pd.Timestamp(a, 1, 1))] for a in MARCOS]

    def cagr(i, f):
        return ((s.loc[f] / s.loc[i]) ** (365.25 / (f - i).days) - 1) * 100

    out = {j: cagr(i, f) for j, i, f in zip(janelas_bienais(), pts, pts[1:])}
    out["período inteiro"] = cagr(s.index[0], s.index[-1])
    return out, pts


def series_painel(v, custo):
    """series de um painel (K = 9): as 5 frequencias, ibov e cdi"""
    out = {}
    for freq in FREQS:
        col = f"K{K_FIG}_{freq}_c{custo:.2%}"
        if col in v:
            out[freq] = v[col]
    out["Ibovespa"], out["CDI"] = v["Ibovespa"], v["CDI"]
    return out


def cor_serie(nome):
    return COR_FREQ.get(nome, MUDA)


def fmt_pct(x):
    return f"{x:.1f}".replace(".", ",")


def fig_valor_freq_2x2(matplotlib, plt, val, regra, pasta, custo=CUSTO):
    """igual o valor_por_frequencia (K = 9) mas com os 2 metodos nas linhas e as 2
    medidas nas colunas, todas as frequencias. embaixo de cada painel o cagr de cada bienio"""
    fig = plt.figure(figsize=(13, 10.6))
    externo = fig.add_gridspec(2, 2, hspace=0.16, wspace=0.16)
    medidas = [("variancia", "variância"), ("cvar", "CVaR")]
    metodos = [("cplex", "CPLEX"), ("nsga2", "NSGA-II")]
    letra = iter("abcd")
    ref = None
    for i, (metodo, rm) in enumerate(metodos):
        for j, (medida, rmed) in enumerate(medidas):
            gs = externo[i, j].subgridspec(2, 1, height_ratios=[3, 1.3], hspace=0.16)
            ax = fig.add_subplot(gs[0], sharex=ref, sharey=ref)
            ref = ref or ax
            tab = fig.add_subplot(gs[1], sharex=ref)
            v = val.get((regra, metodo, medida))
            ax.set_title(f"({next(letra)}) {rm} · {rmed}", fontsize=10, color=TINTA, loc="left")
            estilo(ax)
            tab.axis("off")
            if v is None:
                ax.text(0.5, 0.5, "execução pendente", transform=ax.transAxes, ha="center", color=MUDA)
                continue
            series = series_painel(v, custo)
            for nome, s in series.items():
                if nome in FREQS:
                    ax.plot(s.index, s / 1000, color=COR_FREQ[nome], lw=1.2,
                            ls="--" if nome == "sem_rebal" else "-", label=ROT_SERIE[nome])
                else:
                    ax.plot(s.index, s / 1000, color=MUDA, lw=1.0, ls="-" if nome == "Ibovespa" else ":",
                            label=nome)
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0f}"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if j == 0:
                ax.set_ylabel("Valor (R$ mil, escala log)")
            # faixa de cagr, uma coluna por bienio no meio dos dois marcos
            trans = matplotlib.transforms.blended_transform_factory(tab.transData, tab.transAxes)
            linhas = list(series)
            passo = 1 / (len(linhas) + 1)
            pts = None
            for r, nome in enumerate(linhas):
                c, pts = cagr_bienal(series[nome])
                y = 1 - (r + 1.5) * passo
                tab.text(-0.01, y, ROT_CURTO[nome], transform=tab.transAxes, ha="right", va="center",
                         fontsize=7, color=cor_serie(nome))
                for k, jan in enumerate(janelas_bienais()):
                    meio = pts[k] + (pts[k + 1] - pts[k]) / 2
                    tab.text(meio, y, fmt_pct(c[jan]), transform=trans, ha="center", va="center",
                             fontsize=7, color=cor_serie(nome) if nome in FREQS else TINTA2)
            tab.text(-0.01, 1 - 0.5 * passo, "CAGR (% a.a.)", transform=tab.transAxes, ha="right",
                     va="center", fontsize=7, color=TINTA, fontweight="bold")
            for k, jan in enumerate(janelas_bienais()):
                meio = pts[k] + (pts[k + 1] - pts[k]) / 2
                tab.text(meio, 1 - 0.5 * passo, jan, transform=trans, ha="center", va="center",
                         fontsize=7, color=TINTA)
            for pt in pts:
                tab.axvline(pt, color=GRADE_COR, lw=0.6)
            tab.axhline(1 - passo, color=GRADE_COR, lw=0.6)
            ax.tick_params(labelbottom=True)
            if j == 1:
                ax.tick_params(labelleft=False)
    h, r = [], []
    for ax in fig.axes:
        hh, rr = ax.get_legend_handles_labels()
        if len(rr) > len(r):
            h, r = hh, rr
    fig.legend(h, r, frameon=False, fontsize=8.5, ncol=7, loc="lower center")
    titulo(fig, f"Valor da carteira por frequência de rebalanceamento — {REGRAS[regra]}, K = {K_FIG}, "
                f"custo de transação = {custo:.2%}".replace(".", ","))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.06)
    pasta.mkdir(parents=True, exist_ok=True)
    nome = "valor_por_frequencia_2x2" + ("" if custo == 0 else "_com_custo")
    for ext in ("png", "pdf"):
        fig.savefig(pasta / f"{nome}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def tabela_cagr_bienal(val):
    """cagr por bienio das series do valor_por_frequencia_2x2 (K = 9), em md e csv"""
    linhas = []
    for regra in REGRAS:
        for (metodo, medida), rot in COMBOS.items():
            v = val.get((regra, metodo, medida))
            if v is None:
                continue
            for custo in (CUSTO, 0.001):
                for nome, s in series_painel(v, custo).items():
                    c, _ = cagr_bienal(s)
                    linhas.append(dict(cenario=REGRAS[regra], combinacao=rot, custo=custo,
                                       serie=ROT_SERIE[nome], **c))
    df = pd.DataFrame(linhas)
    DEST.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEST / "cagr_bienal.csv", index=False)
    cols = janelas_bienais() + ["período inteiro"]
    md = ["# CAGR por biênio — figuras valor_por_frequencia_2x2 (K = 9)", "",
          "Gerado por `scripts/comparar_grade.py`. CAGR (% a.a.) = (V_fim / V_ini)^(1/anos) − 1, com "
          "anos = dias corridos / 365,25. Os marcos são o primeiro pregão de cada ano par "
          "(2014-01-02, 2016, ..., 2026), os mesmos do eixo x das figuras; \"período inteiro\" vai de "
          "2014-01-02 a 2026-10-07. Ibovespa e CDI não dependem da configuração e se repetem em "
          "todas as tabelas, só como referência.", ""]
    for cenario in df.cenario.unique():
        for custo in (CUSTO, 0.001):
            md += [f"## {cenario} — custo de transação {custo:.2%}".replace(".", ","), ""]
            d = df[(df.cenario == cenario) & (df.custo == custo)]
            for comb in d.combinacao.unique():
                md += [f"### {comb}", "", "| Série | " + " | ".join(cols) + " |",
                       "|---|" + "--:|" * len(cols)]
                for _, x in d[d.combinacao == comb].iterrows():
                    md.append(f"| {x.serie} | " + " | ".join(fmt_pct(x[c]) for c in cols) + " |")
                md.append("")
            # lado a lado, uma linha por (combinacao, frequencia), sem as referencias
            md += [f"### Comparação das 4 combinações — {cenario}, custo {custo:.2%}".replace(".", ","), "",
                   "| Combinação | Frequência | " + " | ".join(cols) + " |",
                   "|---|---|" + "--:|" * len(cols)]
            for freq in FREQS:
                for comb in d.combinacao.unique():
                    x = d[(d.combinacao == comb) & (d.serie == ROT_SERIE[freq])]
                    if len(x):
                        x = x.iloc[0]
                        md.append(f"| {comb} | {x.serie} | " + " | ".join(fmt_pct(x[c]) for c in cols) + " |")
            md.append("")
    (DEST / "cagr_bienal.md").write_text("\n".join(md), encoding="utf-8")


def fig_melhor_cenario(plt, met, bench, pasta, n=15):
    """ranking das configs (cenario, metodo, medida, frequencia) pelo sharpe medio
    nos 7 K, com o min e max entre os K"""
    d = met[met.custo == CUSTO].groupby(["regra", "metodo", "medida", "freq"]).sharpe_cdi.agg(
        ["mean", "min", "max"]).reset_index().sort_values("mean", ascending=False).head(n)[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.36 * len(d) + 1.4))
    y = np.arange(len(d))
    for k, r in enumerate(d.itertuples()):
        c = COR[(r.metodo, r.medida)]
        ax.hlines(k, r.min, r.max, color=c, lw=1, alpha=0.5)
        ax.scatter(r.mean, k, color=c, marker=MARCA_REGRA[r.regra], s=48, zorder=3,
                   facecolors=c if r.medida == "variancia" else "white")
    ax.set_yticks(y, [f"{REGRAS[r.regra]} · {COMBOS[(r.metodo, r.medida)]} · {ROT_FREQ[r.freq]}"
                      for r in d.itertuples()], fontsize=8)
    ib = bench.at["Ibovespa", "sharpe_cdi"]
    ax.axvline(ib, color=TINTA2, lw=1)
    ax.annotate("Ibovespa", (ib, len(d) - 0.5), xytext=(3, 0), textcoords="offset points", fontsize=8,
                color=TINTA2)
    ax.set_xlabel("Sharpe realizado contra o CDI (ponto = média nos 7 K; traço = mín.-máx. entre os K)")
    estilo(ax)
    titulo(fig, f"As {n} melhores configurações — 2014-2026, sem custos (● máximo Sharpe, ▲ risco mínimo; "
                f"vazado = CVaR)")
    salvar(fig, pasta, "ranking_configuracoes")
    plt.close(fig)


def fig_tempo(plt, dec, met, regra, pasta):
    """tempo por decisao (por K) e sharpe medio x tempo medio"""
    d = dec[dec.regra == regra]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    larg = 0.8 / len(COMBOS)
    for i, ((metodo, medida), nome) in enumerate(COMBOS.items()):
        x = d[(d.metodo == metodo) & (d.medida == medida)].groupby("K").tempo_total_s.mean().reindex(KS)
        ax.bar(np.arange(len(KS)) + (i - (len(COMBOS) - 1) / 2) * larg, x, larg * 0.92,
               color=COR[(metodo, medida)], label=nome)
    ax.set_yscale("log")
    ax.set_xticks(range(len(KS)), [str(k) for k in KS])
    ax.set_xlabel("K (nº de ativos)")
    ax.set_ylabel("Tempo por decisão (s, escala log)")
    ax.set_title("(a) Tempo de otimização por data de decisão", fontsize=10, color=TINTA, loc="left")
    ax.legend(frameon=False, fontsize=8)
    estilo(ax)
    m = met[(met.custo == CUSTO) & (met.regra == regra)]
    for (metodo, medida), nome in COMBOS.items():
        t = d[(d.metodo == metodo) & (d.medida == medida)].tempo_total_s.mean()
        sr = m[(m.metodo == metodo) & (m.medida == medida)].sharpe_cdi.mean()
        ax2.scatter(t, sr, s=70, color=COR[(metodo, medida)], zorder=3)
        ax2.annotate(nome, (t, sr), xytext=(6, 4), textcoords="offset points", fontsize=8, color=TINTA2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Tempo médio por decisão (s, escala log)")
    ax2.set_ylabel("Sharpe realizado (média nos K e frequências)")
    ax2.set_title("(b) Qualidade realizada × custo computacional", fontsize=10, color=TINTA, loc="left")
    estilo(ax2)
    titulo(fig, f"{REGRAS[regra]} — decisões mensais (154 datas, 2014-2026)")
    salvar(fig, pasta, "tempo")
    plt.close(fig)


def fig_ativos(plt, pes, regra, pasta, K=K_FIG):
    """quantas vezes cada ativo aparece nas decisoes mensais, por combinacao"""
    p = pes[(pes.regra == regra) & (pes.K == K)]
    cols = {}
    for (metodo, medida), nome in COMBOS.items():
        q = p[(p.metodo == metodo) & (p.medida == medida)]
        if len(q):
            cols[nome] = q.groupby("ticker").t.nunique() / q.t.nunique()
    freq = pd.DataFrame(cols).fillna(0)
    top = freq.mean(axis=1).sort_values(ascending=False).index[:25]
    v = freq.loc[top].values * 100
    fig, ax = plt.subplots(figsize=(6.2, 0.27 * len(top) + 1.6))
    ax.imshow(v, aspect="auto", cmap="Blues", vmin=0, vmax=max(v.max(), 1))
    for (a, b), x in np.ndenumerate(v):
        if x >= 0.5:
            ax.text(b, a, f"{x:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if x > 0.6 * v.max() else TINTA)
    ax.set_xticks(range(len(freq.columns)), [n.replace(" · ", "\n") for n in freq.columns], fontsize=8)
    ax.set_yticks(range(len(top)), top, fontsize=8)
    ax.tick_params(length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.set_title(f"{REGRAS[regra]} — presença do ativo nas decisões mensais (%), K = {K}", fontsize=9.5,
                 color=TINTA, loc="left")
    salvar(fig, pasta, f"ativos_K{K}")
    plt.close(fig)


def efeitos(met):
    """diferencas pareadas de sharpe (mesmo K, frequencia e custo 0) pra cada fator"""
    d = met[met.custo == CUSTO].set_index(["regra", "metodo", "medida", "freq", "K"]).sharpe_cdi
    linhas = []
    for (regra, metodo, medida, freq, K), v in d.items():
        if metodo == "cplex":
            o = d.get((regra, "nsga2", medida, freq, K))
            if o is not None:
                linhas.append({"fator": "metodo", "grupo": f"{REGRAS[regra]}\n{medida}", "freq": freq, "K": K,
                               "dif": v - o})
        if medida == "cvar":
            o = d.get((regra, metodo, "variancia", freq, K))
            if o is not None:
                linhas.append({"fator": "medida", "grupo": f"{REGRAS[regra]}\n{metodo}", "freq": freq, "K": K,
                               "dif": v - o})
        if regra == "max_sharpe":
            o = d.get(("var_min", metodo, medida, freq, K))
            if o is not None:
                linhas.append({"fator": "regra", "grupo": f"{COMBOS[(metodo, medida)]}".replace(" · ", "\n"),
                               "freq": freq, "K": K, "dif": v - o})
    return pd.DataFrame(linhas)


def fig_efeitos(plt, ef, pasta):
    """quanto cada fator muda o sharpe, separado pelos outros fatores"""
    paineis = [("metodo", "(a) Método: CPLEX − NSGA-II"), ("medida", "(b) Medida: CVaR − variância"),
               ("regra", "(c) Cenário: máximo Sharpe − risco mínimo")]
    cores_f = dict(zip(FREQS, ["#52514e", "#2a78d6", "#1baf7a", "#eda100", "#eb6834"]))
    fig, eixos = plt.subplots(1, 3, figsize=(13.5, 4.6))
    rng = np.random.default_rng(0)
    for ax, (fator, tit) in zip(eixos, paineis):
        e = ef[ef.fator == fator]
        grupos = list(dict.fromkeys(e.grupo))
        for i, g in enumerate(grupos):
            x = e[e.grupo == g]
            for f in FREQS:
                y = x[x.freq == f].dif
                ax.scatter(i + rng.uniform(-0.18, 0.18, len(y)), y, s=11, color=cores_f[f], alpha=0.75,
                           label=ROT_FREQ[f] if i == 0 else None, zorder=2)
            ax.hlines(x.dif.mean(), i - 0.3, i + 0.3, color=TINTA, lw=2, zorder=3)
            ax.annotate(f"{x.dif.mean():+.2f}".replace(".", ","), (i + 0.32, x.dif.mean()), fontsize=8,
                        va="center", color=TINTA)
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_xticks(range(len(grupos)), [g.replace("variancia", "variância").replace("cvar", "CVaR")
                                           .replace("cplex", "CPLEX").replace("nsga2", "NSGA-II") for g in grupos],
                      fontsize=8)
        ax.set_xlim(-0.6, len(grupos) - 0.2)
        ax.set_title(tit, fontsize=10, color=TINTA, loc="left")
        estilo(ax)
    eixos[0].set_ylabel("Diferença de Sharpe realizado")
    h, r = eixos[0].get_legend_handles_labels()
    fig.legend(h, r, frameon=False, fontsize=8.5, ncol=5, loc="lower center", title="Rebalanceamento",
               title_fontsize=8)
    titulo(fig, "Efeito de cada escolha no Sharpe realizado — cada ponto é um par (K, frequência) com "
                "o resto fixo; traço = média")
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    pasta.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(pasta / f"efeitos_pareados.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def fig_ex_ante(plt, dec, met, pasta):
    """o q o otimizador prometia (ex-ante) x o q aconteceu (mensal)"""
    fig, eixos = plt.subplots(1, 2, figsize=(11.5, 4.6))
    specs = [("max_sharpe", "sharpe_esp", "sharpe_cdi", 1, "Sharpe", "(a) Máximo Sharpe: Sharpe"),
             ("var_min", "vol_esp", "volatilidade_anual", 100, "Volatilidade (%)",
              "(b) Risco mínimo: volatilidade")]
    m = met[(met.custo == CUSTO) & (met.freq == "mensal")]
    for ax, (regra, c_ea, c_re, esc, eixo, tit) in zip(eixos, specs):
        lim = []
        for (metodo, medida), nome in COMBOS.items():
            a = dec[(dec.regra == regra) & (dec.metodo == metodo) & (dec.medida == medida)].groupby("K")[c_ea].mean()
            b = m[(m.regra == regra) & (m.metodo == metodo) & (m.medida == medida)].set_index("K")[c_re]
            j = a.index.intersection(b.index)
            ax.scatter(a[j] * esc, b[j] * esc, color=COR[(metodo, medida)], marker=MARCA[metodo], s=34,
                       label=nome, zorder=3, facecolors="none" if medida == "cvar" else COR[(metodo, medida)])
            lim += list(a[j] * esc) + list(b[j] * esc)
        lo, hi = min(lim), max(lim)
        ax.plot([lo, hi], [lo, hi], color=MUDA, lw=1, ls=":")
        ax.annotate("realizado = prometido", (hi, hi), xytext=(-4, -12), textcoords="offset points",
                    ha="right", fontsize=7.5, color=MUDA)
        ax.set_xlabel(f"{eixo} ex-ante (média das decisões mensais)")
        ax.set_ylabel(f"{eixo} realizado (mensal, sem custos)")
        ax.set_title(tit, fontsize=10, color=TINTA, loc="left")
        estilo(ax)
    eixos[0].legend(frameon=False, fontsize=8)
    titulo(fig, "Prometido × realizado — cada ponto é um K; vazado = CVaR")
    salvar(fig, pasta, "ex_ante_x_realizado")
    plt.close(fig)


def sobreposicao(pes):
    linhas = []
    for regra in REGRAS:
        for medida in ("variancia", "cvar"):
            for K in KS:
                s = {m: pes[(pes.regra == regra) & (pes.medida == medida) & (pes.metodo == m) & (pes.K == K)]
                     .groupby("t").ticker.agg(frozenset) for m in ("cplex", "nsga2")}
                comum = s["cplex"].index.intersection(s["nsga2"].index)
                if len(comum):
                    j = [len(s["cplex"][t] & s["nsga2"][t]) / len(s["cplex"][t] | s["nsga2"][t]) for t in comum]
                    linhas.append({"regra": regra, "medida": medida, "K": K, "datas": len(comum),
                                   "jaccard_medio": np.mean(j)})
    return pd.DataFrame(linhas)


def fig_sobreposicao(plt, sob, pasta):
    """cplex e nsga2 escolhem os mesmos ativos?"""
    if sob.empty:      # grade parcial, sem cplex ainda
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cores = {("max_sharpe", "variancia"): "#2a78d6", ("max_sharpe", "cvar"): "#2a78d6",
             ("var_min", "variancia"): "#eb6834", ("var_min", "cvar"): "#eb6834"}
    for (regra, medida), g in sob.groupby(["regra", "medida"]):
        x = g.set_index("K").reindex(KS)
        ax.plot(range(len(KS)), x.jaccard_medio, color=cores[(regra, medida)], ls=ESTILO[medida],
                marker=MARCA_REGRA[regra], ms=5, lw=1.5,
                label=f"{REGRAS[regra]} · {'variância' if medida == 'variancia' else 'CVaR'}")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(len(KS)), [str(k) for k in KS])
    ax.set_xlabel("K (nº de ativos)")
    ax.set_ylabel("Sobreposição média (Jaccard)")
    ax.legend(frameon=False, fontsize=8)
    estilo(ax)
    titulo(fig, "Ativos em comum entre CPLEX e NSGA-II na mesma data (1 = mesmos ativos)")
    salvar(fig, pasta, "sobreposicao_cplex_nsga2")
    plt.close(fig)


def fig_tempo_geral(plt, dec, pasta):
    """tempo por decisao das 8 configs"""
    t = dec.groupby(["regra", "metodo", "medida"]).agg(tempo=("tempo_total_s", "mean"),
                                                       res=("n_resolucoes", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    larg = 0.8 / len(COMBOS)
    for i, ((metodo, medida), nome) in enumerate(COMBOS.items()):
        x = t[(t.metodo == metodo) & (t.medida == medida)].set_index("regra").reindex(list(REGRAS))
        pos = np.arange(len(REGRAS)) + (i - (len(COMBOS) - 1) / 2) * larg
        ax.bar(pos, x.tempo, larg * 0.92, color=COR[(metodo, medida)], label=nome)
        for p, v in zip(pos, x.tempo):
            ax.annotate(f"{v:.1f}".replace(".", ","), (p, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7.5, color=TINTA2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(REGRAS)), list(REGRAS.values()))
    ax.set_ylabel("Tempo médio por decisão (s, escala log)")
    ax.legend(frameon=False, fontsize=8)
    estilo(ax)
    titulo(fig, "Custo computacional por decisão (média dos 7 K, 154 datas mensais)")
    salvar(fig, pasta, "tempo_geral")
    plt.close(fig)


# tabelas
def tabelas(met, dec, ef, sob, bench):
    DEST.mkdir(parents=True, exist_ok=True)
    cols = ["regra", "metodo", "medida", "freq", "K", "custo", "retorno_anual", "volatilidade_anual",
            "sharpe_cdi", "drawdown_max", "turnover_anual", "valor_final"]
    met[cols].sort_values(cols[:6]).to_csv(DEST / "metricas.csv", index=False)
    ef.to_csv(DEST / "efeitos_pareados.csv", index=False)
    sob.to_csv(DEST / "sobreposicao.csv", index=False)
    tempo = dec.groupby(["regra", "metodo", "medida", "K"]).agg(
        decisoes=("t", "nunique"), tempo_por_decisao_s=("tempo_total_s", "mean"),
        resolucoes_por_decisao=("n_resolucoes", "mean"), todos_otimos=("todos_otimos", "mean"),
        gap_max=("gap_max", "max")).reset_index()
    tempo.to_csv(DEST / "tempo.csv", index=False)

    def f(x, pct=False):
        return f"{x * 100:.1f}%" if pct else f"{x:.2f}"

    rot = lambda r: f"{REGRAS[r.regra]} · {COMBOS[(r.metodo, r.medida)]}"   # noqa: E731
    out = ["# Resumo da grade completa (D34)", "",
           "Gerado por `scripts/comparar_grade.py`. Σ amostral, base Yahoo + B3, 2014-01-02 a 2026-10-07. "
           "Métricas realizadas (fora da amostra). Os números são lidos daqui para o texto; "
           "nenhuma conclusão é tirada neste arquivo.", "",
           f"Referências: Ibovespa Sharpe {bench.at['Ibovespa', 'sharpe_cdi']:.2f}, "
           f"retorno {bench.at['Ibovespa', 'retorno_anual']:.1%} a.a., "
           f"vol. {bench.at['Ibovespa', 'volatilidade_anual']:.1%}, "
           f"drawdown {bench.at['Ibovespa', 'drawdown_max']:.1%}; "
           f"CDI retorno {bench.at['CDI', 'retorno_anual']:.1%} a.a.", ""]
    for custo in (0.0, 0.001):
        d = met[met.custo == custo]
        g = d.groupby(["regra", "metodo", "medida", "freq"]).agg(
            sharpe=("sharpe_cdi", "mean"), ret=("retorno_anual", "mean"), vol=("volatilidade_anual", "mean"),
            dd=("drawdown_max", "mean"), to=("turnover_anual", "mean")).reset_index()
        out += [f"## Média nos 7 K — custo {custo:.2%}", "",
                "| Configuração | Freq. | Sharpe | Retorno a.a. | Vol. | Drawdown máx. | Turnover a.a. |",
                "|---|---|---|---|---|---|---|"]
        for r in g.sort_values(["regra", "metodo", "medida"]).itertuples():
            out.append(f"| {rot(r)} | {ROT_FREQ[r.freq]} | {f(r.sharpe)} | {f(r.ret, 1)} | {f(r.vol, 1)} | "
                       f"{f(r.dd, 1)} | {f(r.to)} |")
        out += ["", f"## 15 melhores (configuração, K, frequência) por Sharpe — custo {custo:.2%}", "",
                "| # | Configuração | K | Freq. | Sharpe | Retorno a.a. | Vol. | Drawdown máx. |",
                "|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(d.sort_values("sharpe_cdi", ascending=False).head(15).itertuples(), 1):
            out.append(f"| {i} | {rot(r)} | {r.K} | {ROT_FREQ[r.freq]} | {f(r.sharpe_cdi)} | "
                       f"{f(r.retorno_anual, 1)} | {f(r.volatilidade_anual, 1)} | {f(r.drawdown_max, 1)} |")
        out.append("")
    out += ["## Efeitos pareados no Sharpe (custo 0) — média, mínimo, máximo e % de pares > 0", "",
            "| Fator | Grupo | Média | Mín. | Máx. | % > 0 |", "|---|---|---|---|---|---|"]
    for (fat, gr), x in ef.groupby(["fator", "grupo"], sort=False):
        out.append(f"| {fat} | {gr.replace(chr(10), ' · ')} | {x.dif.mean():+.3f} | {x.dif.min():+.3f} | "
                   f"{x.dif.max():+.3f} | {(x.dif > 0).mean():.0%} |")
    out += ["", "## Tempo por decisão (s, média dos K, decisões mensais) e qualidade da solução do CPLEX", "",
            "| Configuração | Tempo (s) | Resoluções/decisão | Decisões com todas ótimas | Gap máx. |",
            "|---|---|---|---|---|"]
    t = tempo.groupby(["regra", "metodo", "medida"]).agg(tp=("tempo_por_decisao_s", "mean"),
                                                        rs=("resolucoes_por_decisao", "mean"),
                                                        ot=("todos_otimos", "mean"), gap=("gap_max", "max"))
    for (regra, metodo, medida), r in t.iterrows():
        cpx = metodo == "cplex"
        out.append(f"| {REGRAS[regra]} · {COMBOS[(metodo, medida)]} | {r.tp:.2f} | {r.rs:.0f} | "
                   f"{f'{r.ot:.1%}' if cpx else '—'} | {f'{r.gap:.1e}' if cpx else '—'} |")
    (DEST / "resumo.md").write_text("\n".join(out) + "\n", encoding="utf-8")


# pasta de fotos
LEIA = """# Gráficos de resultados (grade completa, D34)

Gerados por `Projeto/scripts/comparar_grade.py` a partir de `Projeto/results/grade/`.
Σ amostral, base Yahoo + B3, rebalanceamento 2014-2026, K ∈ {3, 4, 5, 9, 15, 20, 30}.
Números exatos: `Projeto/results/grade/comparacao/resumo.md` e `metricas.csv`.
Guia de leitura: `Projeto/docs/GUIA_LEITURA_GRAFICOS.md`, seção 13.

```
cagr_bienal.md / .csv    CAGR por biênio (marcos do eixo x) das séries de valor_por_frequencia_2x2
0_consolidacao/          visão geral (1-2 figuras para abrir os resultados)
  mapa_sharpe            Sharpe médio nos K: cenário x combinação x frequência
  risco_retorno          retorno x volatilidade realizados de todas as configurações
1_maximo_sharpe/         cenário de retorno (carteira de máximo Sharpe ex-ante)
2_risco_minimo/          cenário de risco (carteira de risco mínimo com K ativos)
  comparacao/            as 4 combinações juntas
    valor_por_frequencia_2x2  valor (K = 9) por frequência, inclusive mensal; painéis =
                              CPLEX/NSGA-II x variância/CVaR (sem custo; _com_custo = 0,10%),
                              com o CAGR de cada biênio embaixo de cada painel
    metricas_por_frequencia   Sharpe, vol. e drawdown (média nos K) x frequência
    metricas_por_k_<freq>     Sharpe, vol. e drawdown x K, uma por frequência
    valor_acumulado_<freq>    valor (K = 9) e drawdown, uma por frequência
    tempo                     tempo por decisão e Sharpe x tempo
    ativos_K9                 presença dos ativos nas decisões mensais
  cplex_variancia/ cplex_cvar/ nsga2_variancia/ nsga2_cvar/
    valor_acumulado           K = 3, 9, 20, 30, anual
    sharpe_por_k              Sharpe x K por frequência, sem custo e com 0,10%
    valor_por_frequencia      K = 9, todas as frequências
3_cruzamentos/           relações entre os fatores
  efeitos_pareados       quanto método, medida e cenário mudam o Sharpe
  ex_ante_x_realizado    o que o otimizador prometia x o que aconteceu
  sobreposicao_cplex_nsga2  CPLEX e NSGA-II escolhem os mesmos ativos?
  tempo_geral            tempo por decisão das 8 configurações
4_melhor_cenario/
  ranking_configuracoes  as 15 melhores (cenário, método, medida, frequência) pelo Sharpe
5_tempo/                 benchmark controlado (scripts/benchmark_tempo.py, D37): cada
                         decisão sozinha na máquina, CPLEX em 1 thread, 13 datas, 7 K
  tempo_por_k                     tempo por decisão x K (mediana e p10-p90)
  razao_cplex_nsga2               quantas vezes o CPLEX é mais lento que o NSGA-II
  tempo_por_data                  tempo ao longo das datas (N cresce)
  custo_backtest_por_frequencia   minutos de otimização do backtest inteiro
```

Tempo: use as figuras e `Projeto/results/tempo/resumo.md` (D37). Os tempos de
`3_cruzamentos/tempo_geral` e `comparacao/tempo` vêm da grade, medidos com os
7 K em paralelo e parte em execuções anteriores com outra carga (D33/D34),
servem só de ordem de grandeza.

Frequências: sem rebalanceamento (compra em 02/01/2014 e mantém), anual,
semestral, trimestral e mensal. Figuras sem custos de transação; as tabelas
trazem também o custo de 0,10%.
"""


def exportar(regras_pastas):
    if FOTO.exists():
        shutil.rmtree(FOTO)
    FOTO.mkdir()
    (FOTO / "README.md").write_text(LEIA, encoding="utf-8")
    for arq in ("cagr_bienal.md", "cagr_bienal.csv"):
        if (DEST / arq).exists():
            shutil.copy2(DEST / arq, FOTO / arq)
    for origem, destino in [(DEST / "0_consolidacao", FOTO / "0_consolidacao"),
                            (DEST / "3_cruzamentos", FOTO / "3_cruzamentos"),
                            (DEST / "4_melhor_cenario", FOTO / "4_melhor_cenario"),
                            (RAIZ / "results" / "tempo", FOTO / "5_tempo")] + regras_pastas:
        for png in origem.glob("*.png"):
            destino.mkdir(parents=True, exist_ok=True)
            shutil.copy2(png, destino / png.name)
    for i, regra in enumerate(REGRAS, 1):
        base = FOTO / f"{i}_{'maximo_sharpe' if regra == 'max_sharpe' else 'risco_minimo'}"
        for (metodo, medida) in COMBOS:
            o = GRADE / cfg(regra, metodo, medida)
            for g in ("valor_acumulado", "sharpe_por_k", "valor_por_frequencia"):
                if (o / f"{g}.png").exists():
                    (base / f"{metodo}_{medida}").mkdir(parents=True, exist_ok=True)
                    shutil.copy2(o / f"{g}.png", base / f"{metodo}_{medida}" / f"{g}.png")


def main():
    met, dec, pes, val, bench = carregar()
    matplotlib, plt = preparar_plt()
    ef, sob = efeitos(met), sobreposicao(pes)
    tabelas(met, dec, ef, sob, bench)
    tabela_cagr_bienal(val)

    fig_mapa_sharpe(plt, met, bench, DEST / "0_consolidacao")
    fig_risco_retorno(plt, met, bench, DEST / "0_consolidacao")
    pares = []
    for i, regra in enumerate(REGRAS, 1):
        p = DEST / regra
        fig_por_frequencia(plt, met, bench, regra, p)
        for freq in FREQS:
            fig_por_k(plt, met, bench, regra, freq, p)
            fig_valor(matplotlib, plt, val, regra, freq, p)
        fig_valor_freq_2x2(matplotlib, plt, val, regra, p)
        fig_valor_freq_2x2(matplotlib, plt, val, regra, p, custo=0.001)
        fig_tempo(plt, dec, met, regra, p)
        fig_ativos(plt, pes, regra, p)
        pares.append((p, FOTO / f"{i}_{'maximo_sharpe' if regra == 'max_sharpe' else 'risco_minimo'}" / "comparacao"))
    fig_efeitos(plt, ef, DEST / "3_cruzamentos")
    fig_ex_ante(plt, dec, met, DEST / "3_cruzamentos")
    fig_sobreposicao(plt, sob, DEST / "3_cruzamentos")
    fig_tempo_geral(plt, dec, DEST / "3_cruzamentos")
    fig_melhor_cenario(plt, met, bench, DEST / "4_melhor_cenario")
    exportar(pares)
    print((DEST / "resumo.md").read_text(encoding="utf-8")[:6000])
    print(f"\nFiguras em {DEST.relative_to(RAIZ)} e {FOTO}")


if __name__ == "__main__":
    main()
