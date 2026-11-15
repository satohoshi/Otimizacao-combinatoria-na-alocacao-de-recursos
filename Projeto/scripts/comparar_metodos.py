"""comparacao {cplex, nsga2} x {variancia, cvar} numa data de decisao

pra cada K e medida de risco:
  - cplex (exato): risco minimo + grade de R ate o retorno max, igual no fronteira.py
  - nsga2: uma rodada ja da a frente toda, roda com varias sementes pq eh aleatorio
toda solucao eh avaliada nas duas medidas pra ver uma no espaco da outra

indicadores do nsga2 contra o cplex (mesma medida e K):
  gap_risco_min  (risco min nsga - otimo) / otimo
  gap_grade      em cada R_j da grade do cplex, o menor risco do nsga com retorno >= R_j
                 comparado com o otimo em R_j (da >= 0 tirando a tolerancia), media e max
  cobertura      fracao dos R_j q o nsga alcanca
  hv_rel         hipervolume nsga / cplex (o cplex so tem --pontos pontos, entao o hv
                 dele eh um limite inferior e hv_rel > 1 pode acontecer)

uso:
    python3 scripts/comparar_metodos.py                       # t = 2014-01-02, sigma amostral, K = 5, 9, 20
    python3 scripts/comparar_metodos.py --ks 4 9 15 30 --sementes 10
    python3 scripts/comparar_metodos.py --reusar              # so refaz tabelas e graficos

saidas em results/comparacao/<t>_<b3|semb3>_<sigma>/:
    instancia/          assets.csv, covariance.csv, scenarios.csv usados
    solucoes.csv        uma linha por carteira (metodo, medida, K, semente/ponto, mu, var, cvar, tempo...)
    pesos.csv           pesos > 0 de cada carteira (id, ticker, peso)
    execucoes.csv       uma linha por rodada (cplex: a grade toda, nsga2: uma semente)
    resumo.csv, resumo.md   indicadores por combinacao e K
    fronteiras, espacos_cruzados, tempo_qualidade, ativos_K<k> (.png/.pdf)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dados  # noqa: E402
import fronteira  # noqa: E402

RAIZ = dados.RAIZ
EXE = fronteira.EXE
L, U = fronteira.L, fronteira.U
BETA = 0.95
TIME_LIMIT, GAP = fronteira.TIME_LIMIT, fronteira.GAP
METODOS = ("cplex", "nsga2")
MEDIDAS = ("variancia", "cvar")
ROTULO = {("cplex", "variancia"): "CPLEX · variância", ("cplex", "cvar"): "CPLEX · CVaR",
          ("nsga2", "variancia"): "NSGA-II · variância", ("nsga2", "cvar"): "NSGA-II · CVaR"}
COR = {("cplex", "variancia"): "#2a78d6", ("cplex", "cvar"): "#1baf7a",
       ("nsga2", "variancia"): "#eb6834", ("nsga2", "cvar"): "#eda100"}
TINTA, TINTA2, MUDA, GRADE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


# execucao
def rodar(inst, K, medida, metodo="cplex", R=None, threads=1, beta=BETA, l=L, u=U, **ga):
    cmd = [str(EXE), "--assets", str(inst / "assets.csv"), "--cov", str(inst / "covariance.csv"),
           "--cenarios", str(inst / "scenarios.csv"), "--beta", str(beta), "--risco", medida,
           "--metodo", metodo, "--K", str(K), "--l", str(l), "--u", str(u)]
    if metodo == "cplex":
        cmd += ["--time-limit", str(TIME_LIMIT), "--gap", str(GAP), "--threads", str(threads)]
        if R is not None:
            cmd += ["--R", repr(float(R))]
    else:
        cmd += sum(([f"--{k.replace('_', '-')}", str(v)] for k, v in ga.items()), [])
    out = subprocess.run(cmd, capture_output=True, text=True)
    if not out.stdout.strip():
        raise RuntimeError(f"portfolio_opt falhou: {out.stderr.strip()}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def linha(sol, **extra):
    return {**extra, "retorno": sol["retorno"], "variancia": sol["variancia"],
            "vol": np.sqrt(sol["variancia"]), "cvar": sol["cvar"], "var": sol["var"],
            "risco": sol["risco"], "n_ativos": len(sol["pesos"])}


def fronteira_cplex(inst, mu, K, medida, pontos, threads):
    """risco minimo + grade de R (igual fronteira.tracar), devolve (solucoes, pesos, execucao)"""
    base = rodar(inst, K, medida, threads=threads)
    if not base["viavel"]:
        return [], [], None
    grade = np.linspace(base["retorno"], fronteira.retorno_maximo(mu, K, L, U) - 1e-9, pontos)
    sols, pesos = [], []
    for j, R in enumerate(grade):
        r = base if j == 0 else rodar(inst, K, medida, R=R, threads=threads)
        sid = f"cplex_{medida}_K{K}_R{j:02d}"
        if r["viavel"]:
            sols.append(linha(r, id=sid, metodo="cplex", medida=medida, K=K, semente=None, ponto=j,
                              R=None if j == 0 else R, tempo_s=r["tempo_s"], otimo=r["otimo"],
                              gap=r["gap"], nos=r["nos"]))
            pesos += [{"id": sid, "ticker": tk, "peso": p} for tk, p in r["pesos"].items()]
    df = pd.DataFrame(sols)
    ex = {"metodo": "cplex", "medida": medida, "K": K, "semente": None, "tempo_s": df.tempo_s.sum(),
          "n_solucoes": len(df), "todos_otimos": bool(df.otimo.all()), "gap_max_cplex": df.gap.max(),
          "avaliacoes": None, "geracoes": None}
    return sols, pesos, ex


def frente_nsga(inst, K, medida, semente, pop, geracoes):
    r = rodar(inst, K, medida, metodo="nsga2", seed=semente, pop=pop, geracoes=geracoes)
    if not r["viavel"]:
        return [], [], None
    sols, pesos = [], []
    for j, s in enumerate(r["frente"]):
        sid = f"nsga2_{medida}_K{K}_s{semente}_{j:03d}"
        sols.append(linha(s, id=sid, metodo="nsga2", medida=medida, K=K, semente=semente, ponto=j,
                          R=None, tempo_s=r["tempo_s"], otimo=False, gap=None, nos=None))
        pesos += [{"id": sid, "ticker": tk, "peso": p} for tk, p in s["pesos"].items()]
    ex = {"metodo": "nsga2", "medida": medida, "K": K, "semente": semente, "tempo_s": r["tempo_s"],
          "n_solucoes": len(sols), "todos_otimos": None, "gap_max_cplex": None,
          "avaliacoes": r["avaliacoes"], "geracoes": r["geracoes"]}
    return sols, pesos, ex


# indicadores
def hipervolume(risco, ret, ideal, nadir):
    """hv 2d (min risco, max retorno) normalizado em [0, 1]^2, referencia (1,1)"""
    x = (np.asarray(risco) - ideal[0]) / (nadir[0] - ideal[0])
    y = (ideal[1] - np.asarray(ret)) / (ideal[1] - nadir[1])
    pts = sorted(zip(x, y))
    hv, y_ant = 0.0, 1.0
    for xi, yi in pts:
        if xi >= 1 or yi >= y_ant:
            continue
        hv += (1 - xi) * (y_ant - yi)
        y_ant = yi
    return hv


def indicadores(sol, ex):
    """coloca nas rodadas do nsga2 os indicadores contra o cplex"""
    ex = ex.copy()
    for col in ("gap_risco_min", "gap_grade_medio", "gap_grade_max", "cobertura", "hv", "hv_rel"):
        ex[col] = np.nan
    for (medida, K), g in sol.groupby(["medida", "K"]):
        exato = g[g.metodo == "cplex"].sort_values("ponto")
        if exato.empty:
            continue
        ideal = (g.risco.min(), g.retorno.max())
        nadir = (g.risco.max() * 1.05, g.retorno.min() - 0.05 * (g.retorno.max() - g.retorno.min()))
        hv_ex = hipervolume(exato.risco, exato.retorno, ideal, nadir)
        sel = (ex.medida == medida) & (ex.K == K)
        ex.loc[sel & (ex.metodo == "cplex"), "hv"] = hv_ex
        for i in ex.index[sel & (ex.metodo == "nsga2")]:
            f = g[(g.metodo == "nsga2") & (g.semente == ex.at[i, "semente"])]
            gaps, alcancou = [], 0
            for _, p in exato.iterrows():
                ok = f[f.retorno >= p.retorno - 1e-12]
                if len(ok):
                    alcancou += 1
                    gaps.append(ok.risco.min() / p.risco - 1)
            hv = hipervolume(f.risco, f.retorno, ideal, nadir)
            ex.loc[i, ["gap_risco_min", "gap_grade_medio", "gap_grade_max", "cobertura", "hv", "hv_rel"]] = [
                f.risco.min() / exato.risco.iloc[0] - 1, np.mean(gaps), np.max(gaps),
                alcancou / len(exato), hv, hv / hv_ex]
    return ex


def resumo(sol, ex, pesos):
    linhas = []
    for (metodo, medida, K), g in ex.groupby(["metodo", "medida", "K"], sort=False):
        s = sol[(sol.metodo == metodo) & (sol.medida == medida) & (sol.K == K)]
        ids = set(s.id)
        tick = pesos[pesos.id.isin(ids)].ticker
        rmin = s.loc[s.risco.idxmin()]
        d = {"combinacao": ROTULO[(metodo, medida)], "metodo": metodo, "medida": medida, "K": K,
             "execucoes": len(g), "tempo_medio_s": g.tempo_s.mean(), "tempo_dp_s": g.tempo_s.std(),
             "solucoes_por_execucao": g.n_solucoes.mean(),
             "tempo_por_solucao_s": (g.tempo_s / g.n_solucoes).mean(),
             "ativos_distintos": tick.nunique(),
             "risco_minimo": rmin.risco, "retorno_risco_min": rmin.retorno,
             "vol_risco_min": rmin.vol, "cvar_risco_min": rmin.cvar,
             "ativos_risco_min": " ".join(sorted(pesos[pesos.id == rmin.id].ticker))}
        if metodo == "cplex":
            d |= {"todos_otimos": bool(g.todos_otimos.iloc[0]), "gap_max_cplex": g.gap_max_cplex.iloc[0]}
        else:
            for c in ("gap_risco_min", "gap_grade_medio", "gap_grade_max", "cobertura", "hv_rel"):
                d[c] = g[c].mean()
                d[c + "_dp"] = g[c].std()
            d["avaliacoes"] = g.avaliacoes.mean()
        linhas.append(d)
    return pd.DataFrame(linhas)


def semente_mediana(ex, medida, K):
    """semente com hv_rel mediano, eh a rodada tipica q aparece nas figuras"""
    g = ex[(ex.metodo == "nsga2") & (ex.medida == medida) & (ex.K == K)].sort_values("hv_rel")
    return None if g.empty else int(g.semente.iloc[(len(g) - 1) // 2])


# graficos
def estilo(ax):
    ax.grid(color=GRADE, lw=0.6)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def salvar(fig, destino, nome):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(destino / f"{nome}.{ext}", dpi=200, facecolor="white")


def eixo_risco(medida):
    return ("vol", "Volatilidade anualizada (%)") if medida == "variancia" else \
        ("cvar", f"CVaR {BETA:.0%} da perda diária (%)".replace(".", ","))


def serie(sol, ex, metodo, medida, K):
    s = sol[(sol.metodo == metodo) & (sol.medida == medida) & (sol.K == K)]
    if metodo == "nsga2":
        s = s[s.semente == semente_mediana(ex, medida, K)]
    return s.sort_values("risco")


def graficos(sol, ex, res, pesos, destino, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TINTA2,
                         "xtick.color": MUDA, "ytick.color": MUDA})
    ks = sorted(sol.K.unique())

    # fig 1: fronteiras cplex x nsga2, uma linha por medida e uma coluna por K
    fig, eixos = plt.subplots(2, len(ks), figsize=(3.6 * len(ks) + 0.6, 7.2), squeeze=False)
    for i, medida in enumerate(MEDIDAS):
        col, rot = eixo_risco(medida)
        for j, K in enumerate(ks):
            ax = eixos[i, j]
            c = serie(sol, ex, "cplex", medida, K)
            n = serie(sol, ex, "nsga2", medida, K)
            ax.plot(n[col] * 100, n.retorno * 100, "o", ms=3.2, color=COR[("nsga2", medida)], alpha=0.85,
                    label=f"NSGA-II (semente {semente_mediana(ex, medida, K)})", zorder=2)
            ax.plot(c[col] * 100, c.retorno * 100, "-", lw=1.8, color=COR[("cplex", medida)],
                    marker="s", ms=3.5, label="CPLEX (ótimo por ponto)", zorder=3)
            ax.set_title(f"({'abcdefgh'[i * len(ks) + j]}) {'Variância' if medida == 'variancia' else 'CVaR'}, "
                         f"K = {K}", fontsize=10, color=TINTA, loc="left")
            ax.set_xlabel(rot)
            if j == 0:
                ax.set_ylabel("Retorno esperado anualizado (%)")
            estilo(ax)
            if j == 0:
                ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Fronteiras do CPLEX e do NSGA-II — {titulo}", fontsize=10, color=TINTA, x=0.01, ha="left")
    salvar(fig, destino, "fronteiras")
    plt.close(fig)

    # fig 2: as 4 combinacoes nos dois espacos de risco (K do meio)
    K = ks[len(ks) // 2]
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, medida_eixo in zip(eixos, MEDIDAS):
        col, rot = eixo_risco(medida_eixo)
        for metodo in METODOS:
            for medida in MEDIDAS:
                # ordena pelo retorno pq no espaco da outra medida a curva n eh monotona
                s = serie(sol, ex, metodo, medida, K).sort_values("retorno")
                if metodo == "cplex":
                    ax.plot(s[col] * 100, s.retorno * 100, "-", lw=1.6, color=COR[(metodo, medida)],
                            marker="s", ms=3.5, label=ROTULO[(metodo, medida)], zorder=3)
                else:
                    ax.plot(s[col] * 100, s.retorno * 100, "o", ms=3, alpha=0.75,
                            color=COR[(metodo, medida)], label=ROTULO[(metodo, medida)], zorder=2)
        ax.set_xlabel(rot)
        ax.set_ylabel("Retorno esperado anualizado (%)")
        estilo(ax)
    eixos[0].set_title("(a) Espaço retorno × volatilidade", fontsize=10, color=TINTA, loc="left")
    eixos[1].set_title(f"(b) Espaço retorno × CVaR {BETA:.0%}".replace(".", ","), fontsize=10, color=TINTA,
                       loc="left")
    eixos[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Carteiras otimizadas por uma medida, avaliadas nas duas — K = {K}, {titulo}",
                 fontsize=10, color=TINTA, x=0.01, ha="left")
    salvar(fig, destino, "espacos_cruzados")
    plt.close(fig)

    # fig 3: (a) tempo por combinacao e K, (b) gap medio do nsga2 na grade
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    combos = [(m, r) for m in METODOS for r in MEDIDAS]
    larg = 0.8 / len(combos)
    for k, cb in enumerate(combos):
        d = res[(res.metodo == cb[0]) & (res.medida == cb[1])].set_index("K").reindex(ks)
        x = np.arange(len(ks)) + (k - (len(combos) - 1) / 2) * larg
        ax.bar(x, d.tempo_medio_s, larg * 0.92, color=COR[cb], label=ROTULO[cb],
               yerr=d.tempo_dp_s.fillna(0) if cb[0] == "nsga2" else None, capsize=2,
               error_kw={"elinewidth": 0.8, "ecolor": TINTA2})
    ax.set_yscale("log")
    ax.set_xticks(range(len(ks)), [str(k) for k in ks])
    ax.set_xlabel("K (nº de ativos)")
    ax.set_ylabel("Tempo de parede (s, escala log)")
    ax.set_title("(a) Tempo para obter a fronteira", fontsize=10, color=TINTA, loc="left")
    ax.legend(frameon=False, fontsize=8)
    estilo(ax)
    for medida, mk in zip(MEDIDAS, ("o", "s")):
        d = res[(res.metodo == "nsga2") & (res.medida == medida)].set_index("K").reindex(ks)
        ax2.errorbar(range(len(ks)), d.gap_grade_medio * 100, yerr=d.gap_grade_medio_dp.fillna(0) * 100,
                     marker=mk, ms=5, lw=1.4, capsize=3, color=COR[("nsga2", medida)],
                     label=f"NSGA-II · {'variância' if medida == 'variancia' else 'CVaR'}")
    ax2.axhline(0, color="#c3c2b7", lw=0.8)
    ax2.set_xticks(range(len(ks)), [str(k) for k in ks])
    ax2.set_xlabel("K (nº de ativos)")
    ax2.set_ylabel("Excesso de risco sobre o ótimo (%)\n(em variância ou em CVaR)")
    ax2.set_title("(b) Gap médio do NSGA-II nos pontos da grade do CPLEX", fontsize=10, color=TINTA, loc="left")
    ax2.legend(frameon=False, fontsize=8)
    estilo(ax2)
    fig.suptitle(f"Custo computacional e qualidade — {titulo}", fontsize=10, color=TINTA, x=0.01, ha="left")
    salvar(fig, destino, "tempo_qualidade")
    plt.close(fig)

    # fig 4 (uma por K): (a) pesos da de risco minimo, (b) frequencia na fronteira
    for K in ks:
        cols = [ROTULO[cb] for cb in combos]
        peso_min, freq = {}, {}
        for cb in combos:
            s = serie(sol, ex, cb[0], cb[1], K)
            if s.empty:
                continue
            p = pesos[pesos.id.isin(s.id)]
            peso_min[ROTULO[cb]] = p[p.id == s.loc[s.risco.idxmin(), "id"]].set_index("ticker").peso
            freq[ROTULO[cb]] = p.groupby("ticker").id.nunique() / len(s)
        peso_min, freq = pd.DataFrame(peso_min).reindex(columns=cols), pd.DataFrame(freq).reindex(columns=cols)
        ordem = freq.fillna(0).mean(axis=1).sort_values(ascending=False).index[:30]
        ordem = list(dict.fromkeys(list(peso_min.dropna(how="all").index) + list(ordem)))[:35]
        ordem = sorted(ordem, key=lambda t: -freq.fillna(0).loc[t].mean())
        fig, eixos = plt.subplots(1, 2, figsize=(9.5, 0.26 * len(ordem) + 1.9), sharey=True)
        for ax, m, tit, fmt in [(eixos[0], peso_min, "(a) Pesos da carteira de risco mínimo", "{:.0f}"),
                                (eixos[1], freq, "(b) Presença na fronteira (% das carteiras)", "{:.0f}")]:
            v = m.reindex(ordem).fillna(0).values * 100
            ax.imshow(v, aspect="auto", cmap="Blues", vmin=0, vmax=max(v.max(), 1))
            for (a, b), x in np.ndenumerate(v):
                if x >= 0.5:
                    ax.text(b, a, fmt.format(x), ha="center", va="center", fontsize=6.5,
                            color="white" if x > 0.6 * v.max() else TINTA)
            ax.set_xticks(range(len(cols)), [c.replace(" · ", "\n") for c in cols], fontsize=8)
            ax.set_yticks(range(len(ordem)), ordem, fontsize=7.5)
            ax.set_title(tit, fontsize=10, color=TINTA, loc="left")
            ax.tick_params(length=0)
            for lado in ax.spines.values():
                lado.set_visible(False)
        fig.suptitle(f"Ativos escolhidos por combinação — K = {K}, {titulo}", fontsize=10, color=TINTA,
                     x=0.01, ha="left")
        salvar(fig, destino, f"ativos_K{K}")
        plt.close(fig)


def tabela_md(res, destino, cab):
    def pct(x):
        return "—" if pd.isna(x) else f"{100 * x:.2f}".replace(".", ",")

    def num(x, d=2):
        return "—" if pd.isna(x) else f"{x:.{d}f}".replace(".", ",")
    out = [f"# Comparação de métodos — {cab}", "",
           "| Combinação | K | Tempo (s) | Soluções | Risco mín. | Ativos (risco mín.) | Gap risco mín. (%) "
           "| Gap médio grade (%) | Cobertura (%) | HV rel. |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res.sort_values(["K", "metodo", "medida"]).itertuples():
        t = num(r.tempo_medio_s) + (f" ± {num(r.tempo_dp_s)}" if r.metodo == "nsga2" else "")
        rm = pct(r.vol_risco_min) + " (σ)" if r.medida == "variancia" else pct(r.cvar_risco_min) + " (CVaR)"
        out.append(f"| {r.combinacao} | {r.K} | {t} | {num(r.solucoes_por_execucao, 0)} | {rm} | "
                   f"{r.ativos_risco_min} | {pct(getattr(r, 'gap_risco_min', np.nan))} | "
                   f"{pct(getattr(r, 'gap_grade_medio', np.nan))} | {pct(getattr(r, 'cobertura', np.nan))} | "
                   f"{num(getattr(r, 'hv_rel', np.nan), 3)} |")
    (destino / "resumo.md").write_text("\n".join(out) + "\n")


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", default="2014-01-02")
    ap.add_argument("--sigma", choices=["amostral", "lw"], default="amostral")
    ap.add_argument("--sem-b3", action="store_true")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 9, 20])
    ap.add_argument("--pontos", type=int, default=15, help="pontos da grade de R do CPLEX")
    ap.add_argument("--sementes", type=int, default=5, help="execuções do NSGA-II por (medida, K)")
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--geracoes", type=int, default=500)
    ap.add_argument("--threads", type=int, default=1, help="threads do CPLEX (1 = igual ao NSGA-II)")
    ap.add_argument("--reusar", action="store_true", help="não resolve nada; refaz tabelas e gráficos")
    args = ap.parse_args()
    if not EXE.exists():
        sys.exit("Compile antes: cmake -S . -B build && cmake --build build")

    tag = f"{args.t}_{'semb3' if args.sem_b3 else 'b3'}_{args.sigma}"
    destino = RAIZ / "results" / "comparacao" / tag
    inst = destino / "instancia"
    titulo = f"decisão em {args.t}, Σ {args.sigma}, CVaR β = {BETA:.2f}".replace("0.", "0,")

    if not args.reusar:
        est = dados.estimar(dados.carregar(usar_b3=not args.sem_b3), args.t)
        dados.exportar_instancia(est, inst, usar_lw=(args.sigma == "lw"))
        mu = est.mu[est.tickers].values
        N, S = len(est.tickers), len(est.retornos)
        print(f"N = {N}, S = {S} cenários; CVaR no CPLEX: {2 * N + S + 1} variáveis, "
              f"{2 * N + S + 3} restrições", flush=True)
        sols, pesos, exs = [], [], []
        for K in args.ks:
            for medida in MEDIDAS:
                s, p, e = fronteira_cplex(inst, mu, K, medida, args.pontos, args.threads)
                sols += s
                pesos += p
                if e:
                    exs.append(e)
                    print(f"CPLEX   {medida:9s} K={K:2d}: {e['n_solucoes']} pontos, {e['tempo_s']:7.2f} s, "
                          f"todos ótimos: {e['todos_otimos']}", flush=True)
                for sem in range(1, args.sementes + 1):
                    s, p, e = frente_nsga(inst, K, medida, sem, args.pop, args.geracoes)
                    sols += s
                    pesos += p
                    if e:
                        exs.append(e)
                print(f"NSGA-II {medida:9s} K={K:2d}: {args.sementes} sementes, tempo médio "
                      f"{np.mean([x['tempo_s'] for x in exs[-args.sementes:]]):.2f} s", flush=True)
        sol, pesos, ex = pd.DataFrame(sols), pd.DataFrame(pesos), pd.DataFrame(exs)
        sol.to_csv(destino / "solucoes.csv", index=False)
        pesos.to_csv(destino / "pesos.csv", index=False)
        ex.to_csv(destino / "execucoes_brutas.csv", index=False)
    else:
        sol = pd.read_csv(destino / "solucoes.csv")
        pesos = pd.read_csv(destino / "pesos.csv")
        ex = pd.read_csv(destino / "execucoes_brutas.csv")

    ex = indicadores(sol, ex)
    ex.to_csv(destino / "execucoes.csv", index=False)
    res = resumo(sol, ex, pesos)
    res.to_csv(destino / "resumo.csv", index=False)
    tabela_md(res, destino, titulo)
    graficos(sol, ex, res, pesos, destino, titulo)

    pd.set_option("display.width", 220)
    cols = ["combinacao", "K", "tempo_medio_s", "solucoes_por_execucao", "risco_minimo", "gap_risco_min",
            "gap_grade_medio", "gap_grade_max", "cobertura", "hv_rel", "ativos_distintos"]
    print("\n" + res[[c for c in cols if c in res]].sort_values(["K", "combinacao"])
          .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nSaídas em {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
