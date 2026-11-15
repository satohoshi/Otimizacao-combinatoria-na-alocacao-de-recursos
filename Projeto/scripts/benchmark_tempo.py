"""benchmark de tempo controlado

os tempos da grade n servem pra comparar direito pq parte das decisoes veio de
rodadas com 7 cplex em paralelo, e em cada data os 7 K rodam em threads juntas
aqui cada decisao roda sozinha, uma por vez, com o cplex em 1 thread, nas 13
datas anuais (primeiro pregao de janeiro, 2014-2026), nas 8 configs
{max sharpe, risco min} x {cplex, nsga2} x {variancia, cvar} e nos 7 K
o nsga2 roda com 3 sementes pra ver quanto o tempo varia

medidas por decisao:
    tempo_solver_s   soma dos tempos internos do solver (em volta do cplex.solve()
                     ou do loop do nsga2)
    tempo_parede_s   tempo total da decisao, inclui abrir o processo, ler e montar
    n_resolucoes     quantas vezes chamou o solver (cplex + max sharpe: 25 + 9 = 34,
                     resto: 1)

saidas em results/tempo/ (benchmark.csv, resumo.md, figuras) e copia das figuras
em ../graficos_resultados/5_tempo/

uso:  ../venv/bin/python3 scripts/benchmark_tempo.py            # roda e analisa (continua de onde parou)
      ../venv/bin/python3 scripts/benchmark_tempo.py --analisar # so a analise
"""
import argparse
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest  # noqa: E402
import comparar_metodos  # noqa: E402
import dados  # noqa: E402

RAIZ = backtest.RAIZ
DEST = RAIZ / "results" / "tempo"
FOTO = RAIZ.parent / "graficos_resultados" / "5_tempo"
ARQ = DEST / "benchmark.csv"

REGRAS = {"max_sharpe": "Máximo Sharpe", "var_min": "Risco mínimo"}
COMBOS = {("cplex", "variancia"): "CPLEX · variância", ("cplex", "cvar"): "CPLEX · CVaR",
          ("nsga2", "variancia"): "NSGA-II · variância", ("nsga2", "cvar"): "NSGA-II · CVaR"}
COR = {("cplex", "variancia"): "#2a78d6", ("cplex", "cvar"): "#1baf7a",
       ("nsga2", "variancia"): "#eb6834", ("nsga2", "cvar"): "#eda100"}
ESTILO = {"variancia": "-", "cvar": "--"}
MARCA = {"cplex": "o", "nsga2": "s"}
KS = list(backtest.CARTEIRAS)
SEMENTES = (1, 2, 3)
TINTA, TINTA2, MUDA, GRADE_COR = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# qtd de datas de decisao de cada frequencia (pra estimar o custo do backtest)
N_DATAS = {"sem_rebal": 1, "anual": 13, "semestral": 26, "trimestral": 52, "mensal": 154}


# medicao
def decisao(inst, mu, rf, K, regra, metodo, medida, semente):
    l, u = backtest.CARTEIRAS[K]
    t0 = time.perf_counter()
    if metodo == "nsga2":
        backtest.NSGA_SEMENTE = semente
        r = backtest.nsga_decisao(inst, mu, rf, K, l, u, medida, regra)
    else:
        extra = ["--risco", "cvar", "--cenarios", str(inst / "scenarios.csv"), "--beta",
                 str(comparar_metodos.BETA)] if medida == "cvar" else []
        f = backtest.max_sharpe if regra == "max_sharpe" else backtest.var_minima
        r = f(inst, mu, rf, K, l, u, extra=extra)
    parede = time.perf_counter() - t0
    if r is None:
        return {"viavel": False, "tempo_parede_s": parede}
    return {"viavel": True, "tempo_parede_s": parede, "tempo_solver_s": r["tempo_total_s"],
            "n_resolucoes": r["n_resolucoes"], "todos_otimos": r["todos_otimos"], "gap_max": r["gap_max"],
            "sharpe_esp": r["sharpe_esp"]}


def medir():
    DEST.mkdir(parents=True, exist_ok=True)
    feito = pd.read_csv(ARQ) if ARQ.exists() else pd.DataFrame(columns=["t", "regra", "metodo", "medida", "K",
                                                                         "semente"])
    chave = set(zip(feito.t.astype(str), feito.regra, feito.metodo, feito.medida, feito.K.astype(int),
                    feito.semente.astype(int)))
    brutos, cdi = dados.carregar(usar_b3=True), dados.carregar_cdi()
    datas = backtest.datas_frequencia(brutos.ibov.index, "anual")
    linhas = feito.to_dict("records")
    for n, t in enumerate(datas, 1):
        est = dados.estimar(brutos, t)
        rf = backtest.taxa_livre(cdi, t)
        mu = est.mu[est.tickers].values
        with tempfile.TemporaryDirectory() as tmp:
            inst = Path(tmp)
            dados.exportar_instancia(est, inst)
            for K in KS:
                for regra in REGRAS:
                    for (metodo, medida) in COMBOS:
                        for s in (SEMENTES if metodo == "nsga2" else (0,)):
                            if (str(t.date()), regra, metodo, medida, K, s) in chave:
                                continue
                            r = decisao(inst, mu, rf, K, regra, metodo, medida, s)
                            linhas.append({"t": str(t.date()), "N": len(est.tickers), "regra": regra,
                                           "metodo": metodo, "medida": medida, "K": K, "semente": s, **r})
        pd.DataFrame(linhas).to_csv(ARQ, index=False)
        d = pd.DataFrame(linhas)
        d = d[d.t == str(t.date())]
        print(f"[{n}/{len(datas)}] {t.date()} N = {len(est.tickers)}: " + ", ".join(
            f"{m[0]}·{m[1][:3]}·{r_[:3]} {x:.1f}s" for (r_, *m), x in
            d.groupby(["regra", "metodo", "medida"]).tempo_parede_s.mean().items()), flush=True)


# analise
def preparar_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TINTA2,
                         "xtick.color": MUDA, "ytick.color": MUDA})
    return plt


def estilo(ax):
    ax.grid(color=GRADE_COR, lw=0.6)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def salvar(fig, nome):
    for ext in ("png", "pdf"):
        fig.savefig(DEST / f"{nome}.{ext}", dpi=200, facecolor="white")


def fmt(x, casas=2):
    return f"{x:.{casas}f}".replace(".", ",")


def fig_por_k(plt, d):
    """tempo por decisao x K por cenario: mediana e faixa p10-p90 nas 13 datas"""
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, regra in zip(eixos, REGRAS):
        for (metodo, medida), nome in COMBOS.items():
            x = d[(d.regra == regra) & (d.metodo == metodo) & (d.medida == medida)].groupby("K").tempo_parede_s
            med, lo, hi = x.median().reindex(KS), x.quantile(0.1).reindex(KS), x.quantile(0.9).reindex(KS)
            ax.plot(range(len(KS)), med, color=COR[(metodo, medida)], ls=ESTILO[medida], marker=MARCA[metodo],
                    ms=5, lw=1.6, label=nome)
            ax.fill_between(range(len(KS)), lo, hi, color=COR[(metodo, medida)], alpha=0.12, lw=0)
        ax.set_yscale("log")
        ax.set_xticks(range(len(KS)), [str(k) for k in KS])
        ax.set_xlabel("K (nº de ativos)")
        ax.set_title(f"({'ab'[list(REGRAS).index(regra)]}) {REGRAS[regra]}", fontsize=10, color=TINTA, loc="left")
        estilo(ax)
    eixos[0].set_ylabel("Tempo por decisão (s, escala log)")
    eixos[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Tempo de processamento por decisão — linha = mediana nas 13 datas, faixa = p10-p90; "
                 "execução isolada, 1 thread", fontsize=10, color=TINTA, x=0.01, ha="left")
    fig.tight_layout()
    salvar(fig, "tempo_por_k")
    plt.close(fig)


def fig_razao(plt, d):
    """quantas vezes o cplex eh mais lento q o nsga2 (mesma data, K, cenario e medida)"""
    c = d[d.metodo == "cplex"].set_index(["t", "regra", "medida", "K"]).tempo_parede_s
    n = d[d.metodo == "nsga2"].groupby(["t", "regra", "medida", "K"]).tempo_parede_s.mean()
    r = (c / n).dropna().rename("razao").reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    cores = {("max_sharpe", "variancia"): "#2a78d6", ("max_sharpe", "cvar"): "#2a78d6",
             ("var_min", "variancia"): "#eb6834", ("var_min", "cvar"): "#eb6834"}
    for (regra, medida), g in r.groupby(["regra", "medida"]):
        x = g.groupby("K").razao.median().reindex(KS)
        ax.plot(range(len(KS)), x, color=cores[(regra, medida)], ls=ESTILO[medida],
                marker="o" if regra == "max_sharpe" else "^", ms=5, lw=1.6,
                label=f"{REGRAS[regra]} · {'variância' if medida == 'variancia' else 'CVaR'}")
    ax.axhline(1, color=MUDA, lw=1, ls=":")
    ax.annotate("mesmo tempo", (len(KS) - 1, 1), xytext=(0, 4), textcoords="offset points", ha="right",
                fontsize=8, color=MUDA)
    ax.set_yscale("log")
    ax.set_xticks(range(len(KS)), [str(k) for k in KS])
    ax.set_xlabel("K (nº de ativos)")
    ax.set_ylabel("Tempo CPLEX / tempo NSGA-II (escala log)")
    ax.legend(frameon=False, fontsize=8.5)
    estilo(ax)
    fig.suptitle("Razão de tempo CPLEX / NSGA-II por decisão (mediana nas 13 datas; > 1 = CPLEX mais lento)",
                 fontsize=10, color=TINTA, x=0.01, ha="left")
    fig.tight_layout()
    salvar(fig, "razao_cplex_nsga2")
    plt.close(fig)
    return r


def fig_por_data(plt, d):
    """tempo por decisao ao longo das datas (o N cresce com o tempo)"""
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, regra in zip(eixos, REGRAS):
        for (metodo, medida), nome in COMBOS.items():
            x = d[(d.regra == regra) & (d.metodo == metodo) & (d.medida == medida)].groupby("t").agg(
                tempo=("tempo_parede_s", "mean"), N=("N", "first"))
            ax.plot(pd.to_datetime(x.index), x.tempo, color=COR[(metodo, medida)], ls=ESTILO[medida],
                    marker=MARCA[metodo], ms=4, lw=1.4, label=nome)
        ax.set_yscale("log")
        ax.set_title(f"({'ab'[list(REGRAS).index(regra)]}) {REGRAS[regra]}", fontsize=10, color=TINTA, loc="left")
        estilo(ax)
    n = d.groupby("t").N.first()
    eixos[0].set_ylabel("Tempo por decisão, média nos 7 K (s, log)")
    eixos[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle(f"Tempo por decisão em cada data anual (N = {n.min()} a {n.max()} ativos elegíveis)",
                 fontsize=10, color=TINTA, x=0.01, ha="left")
    fig.tight_layout()
    salvar(fig, "tempo_por_data")
    plt.close(fig)


def fig_projecao(plt, proj):
    """custo de otimizacao do backtest todo (7 K, sequencial) por frequencia"""
    freqs = list(N_DATAS)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    larg = 0.8 / 8
    i = 0
    for regra in REGRAS:
        for (metodo, medida), nome in COMBOS.items():
            x = proj[(proj.regra == regra) & (proj.metodo == metodo) & (proj.medida == medida)].set_index(
                "freq").reindex(freqs).horas
            ax.bar(np.arange(len(freqs)) + (i - 3.5) * larg, x * 60, larg * 0.92, color=COR[(metodo, medida)],
                   hatch="" if regra == "max_sharpe" else "///", edgecolor="white", lw=0,
                   label=f"{REGRAS[regra]} · {nome}")
            i += 1
    ax.set_yscale("log")
    ax.set_xticks(range(len(freqs)), ["sem rebal. (1)", "anual (13)", "semestral (26)", "trimestral (52)",
                                      "mensal (154)"])
    ax.set_ylabel("Minutos de otimização (escala log)")
    ax.set_xlabel("Frequência de rebalanceamento (nº de datas de decisão)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    estilo(ax)
    fig.suptitle("Custo projetado do backtest 2014-2026: nº de datas × 7 K × tempo médio por decisão "
                 "(sem hachura = máximo Sharpe; hachura = risco mínimo)", fontsize=10, color=TINTA, x=0.01,
                 ha="left")
    fig.tight_layout()
    salvar(fig, "custo_backtest_por_frequencia")
    plt.close(fig)


def maquina():
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                             text=True).stdout.strip()
    except OSError:
        cpu = platform.processor()
    return f"{cpu}, {platform.system()} {platform.release()}, Python {platform.python_version()}"


def analisar():
    d = pd.read_csv(ARQ)
    d = d[d.viavel.astype(bool)].copy()
    d["K"] = d.K.astype(int)
    d["tempo_por_resolucao_s"] = d.tempo_solver_s / d.n_resolucoes
    d["sobrecarga_s"] = d.tempo_parede_s - d.tempo_solver_s
    plt = preparar_plt()
    fig_por_k(plt, d)
    razao = fig_razao(plt, d)
    fig_por_data(plt, d)

    g = d.groupby(["regra", "metodo", "medida"])
    tab = g.agg(decisoes=("tempo_parede_s", "size"), media=("tempo_parede_s", "mean"),
                mediana=("tempo_parede_s", "median"), p90=("tempo_parede_s", lambda x: x.quantile(0.9)),
                maximo=("tempo_parede_s", "max"), solver=("tempo_solver_s", "mean"),
                sobrecarga=("sobrecarga_s", "mean"), resol=("n_resolucoes", "mean"),
                por_resol=("tempo_por_resolucao_s", "mean"), otimos=("todos_otimos", "mean"),
                gap=("gap_max", "max")).reset_index()
    porK = d.groupby(["regra", "metodo", "medida", "K"]).tempo_parede_s.median().unstack("K")
    # variacao do nsga2 entre sementes (coef de variacao por data e K)
    ns = d[d.metodo == "nsga2"].groupby(["regra", "medida", "t", "K"]).tempo_parede_s.agg(["mean", "std"])
    cv = (ns["std"] / ns["mean"]).groupby(["regra", "medida"]).mean()
    proj = pd.DataFrame([{"regra": r, "metodo": m, "medida": e, "freq": f,
                          "horas": n * len(KS) * x.tempo_parede_s.mean() / 3600}
                         for (r, m, e), x in g for f, n in N_DATAS.items()])
    fig_projecao(plt, proj)
    tab.to_csv(DEST / "resumo_por_configuracao.csv", index=False)
    razao.to_csv(DEST / "razao_cplex_nsga2.csv", index=False)
    proj.to_csv(DEST / "projecao_backtest.csv", index=False)

    rot = lambda r, m, e: f"{REGRAS[r]} · {COMBOS[(m, e)]}"   # noqa: E731
    out = ["# Benchmark de tempo de processamento (D37)", "",
           "Gerado por `scripts/benchmark_tempo.py`. Cada decisão foi resolvida sozinha na máquina, "
           "uma de cada vez, CPLEX com 1 thread, limite de 60 s e gap de 10⁻⁴ por resolução; NSGA-II com "
           f"população {backtest.NSGA_POP}, {backtest.NSGA_GERACOES} gerações e sementes {SEMENTES}. "
           f"Datas: {d.t.nunique()} (primeiro pregão de janeiro), N = {d.N.min()} a {d.N.max()} ativos; "
           f"K ∈ {KS}.", "", f"Máquina: {maquina()}.", "",
           "Tempo por decisão = relógio de parede da decisão inteira (inclui criar o processo e montar o "
           "modelo). Tempo do resolvedor = soma dos tempos internos de cada chamada. Uma decisão do CPLEX "
           "com máximo Sharpe chama o resolvedor ~34 vezes (25 pontos de R + 9 de refino); as demais, 1 vez "
           "(o NSGA-II devolve a frente inteira numa execução).", "",
           "## Tempo por decisão (s), todas as datas e K", "",
           "| Configuração | Decisões | Média | Mediana | p90 | Máx. | Resolvedor | Sobrecarga | "
           "Resol./decisão | s por resolução | Todas ótimas | Gap máx. |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in tab.itertuples():
        cpx = r.metodo == "cplex"
        out.append(f"| {rot(r.regra, r.metodo, r.medida)} | {r.decisoes} | {fmt(r.media)} | {fmt(r.mediana)} | "
                   f"{fmt(r.p90)} | {fmt(r.maximo)} | {fmt(r.solver)} | {fmt(r.sobrecarga)} | {r.resol:.0f} | "
                   f"{fmt(r.por_resol, 3)} | {f'{r.otimos:.0%}' if cpx else '—'} | "
                   f"{f'{r.gap:.1e}' if cpx else '—'} |")
    out += ["", "## Mediana do tempo por decisão (s) por K", "",
            "| Configuração | " + " | ".join(f"K = {k}" for k in KS) + " |", "|---" * (len(KS) + 1) + "|"]
    for (r, m, e), x in porK.iterrows():
        out.append(f"| {rot(r, m, e)} | " + " | ".join(fmt(x[k]) for k in KS) + " |")
    out += ["", "## Razão CPLEX / NSGA-II (mesma data, K, cenário e medida; NSGA-II = média das sementes)", "",
            "| Cenário · medida | Mediana | Mín. | Máx. | " + " | ".join(f"K = {k}" for k in KS) + " |",
            "|---" * (len(KS) + 4) + "|"]
    for (r, e), x in razao.groupby(["regra", "medida"]):
        k = x.groupby("K").razao.median()
        out.append(f"| {REGRAS[r]} · {'variância' if e == 'variancia' else 'CVaR'} | {fmt(x.razao.median(), 1)} | "
                   f"{fmt(x.razao.min(), 1)} | {fmt(x.razao.max(), 1)} | "
                   + " | ".join(fmt(k.get(kk, np.nan), 1) for kk in KS) + " |")
    out += ["", "## Efeito da medida de risco: tempo CVaR / tempo variância (média das decisões)", "",
            "| Cenário · método | CVaR / variância |", "|---|---|"]
    m = d.groupby(["regra", "metodo", "medida"]).tempo_parede_s.mean()
    for r in REGRAS:
        for met in ("cplex", "nsga2"):
            out.append(f"| {REGRAS[r]} · {'CPLEX' if met == 'cplex' else 'NSGA-II'} | "
                       f"{fmt(m[(r, met, 'cvar')] / m[(r, met, 'variancia')], 1)} |")
    out += ["", "## Variação do tempo do NSGA-II entre sementes (coeficiente de variação médio)", "",
            "| Cenário · medida | CV |", "|---|---|"]
    for (r, e), v in cv.items():
        out.append(f"| {REGRAS[r]} · {'variância' if e == 'variancia' else 'CVaR'} | {v:.1%} |")
    out += ["", "## Custo projetado do backtest 2014-2026 (horas; nº de datas × 7 K × tempo médio)", "",
            "| Configuração | " + " | ".join(f"{f} ({n})" for f, n in N_DATAS.items()) + " |",
            "|---" * (len(N_DATAS) + 1) + "|"]
    for (r, me, e), x in proj.groupby(["regra", "metodo", "medida"], sort=False):
        out.append(f"| {rot(r, me, e)} | " + " | ".join(fmt(h) for h in x.set_index("freq").horas[list(N_DATAS)])
                   + " |")
    (DEST / "resumo.md").write_text("\n".join(out) + "\n", encoding="utf-8")

    FOTO.mkdir(parents=True, exist_ok=True)
    for png in DEST.glob("*.png"):
        shutil.copy2(png, FOTO / png.name)
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analisar", action="store_true", help="só refaz tabelas e figuras a partir do csv")
    a = ap.parse_args()
    if not a.analisar:
        medir()
    analisar()
