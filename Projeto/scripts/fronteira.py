"""fronteira eficiente risco x retorno pra cada K

pra cada K resolve o modelo (build/portfolio_opt) numa grade de R entre o retorno
da carteira de variancia minima e o retorno max viavel. faz tb sem cardinalidade
(K = N, l = 0) pra comparar. ibov e cdi entram como referencia, na mesma janela

uso:
    python3 scripts/fronteira.py                         # t = 2014-01-02, sigma amostral, yahoo + b3
    python3 scripts/fronteira.py --t 2014-01-02 --sigma lw
    python3 scripts/fronteira.py --sem-b3                # so yahoo

le a instancia de data/instancias/<t>_<b3|semb3>/<sigma>/ (preparar_dados.py)
saidas em results/fronteira/<t>_<b3|semb3>_<sigma>/:
    resolucoes.csv   uma linha por resolucao (K, l, u, R, status, risco, retorno, gap, tempo, nos...)
    pesos.csv        pesos > 0 de cada resolucao (id, ticker, peso)
    referencias.csv  ibov, cdi e variancia minima sem cardinalidade
    fronteira.png/.pdf
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

RAIZ = dados.RAIZ
EXE = RAIZ / "build" / "portfolio_opt"

# parametros do modelo
KS = [4, 5, 9, 15, 20, 30]
L, U = 0.01, 0.30
PONTOS_R = 25
TIME_LIMIT, GAP = 60, 1e-4


def resolver(inst, K, l, u, R=None, threads=0, extra=()):
    cmd = [str(EXE), "--assets", str(inst / "assets.csv"), "--cov", str(inst / "covariance.csv"),
           "--K", str(K), "--l", str(l), "--u", str(u),
           "--time-limit", str(TIME_LIMIT), "--gap", str(GAP), "--threads", str(threads)]
    if R is not None:
        cmd += ["--R", repr(float(R))]
    cmd += list(extra)   # ex: --risco cvar --cenarios ...
    out = subprocess.run(cmd, capture_output=True, text=True)
    if not out.stdout.strip():
        raise RuntimeError(f"portfolio_opt falhou: {out.stderr.strip()}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def retorno_maximo(mu, K, l, u):
    """maior mu'w com K ativos em [l, u]: os K de maior mu comecam em l e o resto
    vai, do maior mu pro menor, ate u"""
    m = np.sort(mu)[::-1][:K]
    w = np.full(K, l)
    resto = 1 - K * l
    for i in range(K):
        a = min(u - l, resto)
        w[i] += a
        resto -= a
    return float(m @ w)


def tracar(inst, mu, rotulo, K, l, u, linhas, pesos):
    """variancia minima + grade de R, coloca em `linhas` e `pesos`"""
    base = resolver(inst, K, l, u)
    if not base["viavel"]:
        print(f"{rotulo}: inviável ({base['status']})")
        return
    r_min, r_max = base["retorno"], retorno_maximo(mu, K, l, u)
    # ultimo ponto fica 1e-9 abaixo do R_max pra n depender da tolerancia do solver
    grade = np.linspace(r_min, r_max - 1e-9, PONTOS_R)
    for j, R in enumerate(grade):
        r = base if j == 0 else resolver(inst, K, l, u, R)
        rid = f"{rotulo}_R{j:02d}"
        linhas.append({"id": rid, "serie": rotulo, "K": K, "l": l, "u": u,
                       "R": None if j == 0 else R, "status": r["status"], "viavel": r["viavel"],
                       "otimo": r["otimo"], "variancia": r["risco"], "vol": np.sqrt(r["risco"]),
                       "retorno": r["retorno"], "gap": r["gap"], "limitante": r["limitante"],
                       "tempo_s": r["tempo_s"], "nos": r["nos"], "n_ativos": len(r["pesos"])})
        pesos += [{"id": rid, "ticker": tk, "peso": p} for tk, p in r["pesos"].items()]
    df = pd.DataFrame([x for x in linhas if x["serie"] == rotulo])
    print(f"{rotulo}: R em [{r_min:.4f}, {r_max:.4f}], {df.otimo.sum()}/{len(df)} ótimos, "
          f"tempo total {df.tempo_s.sum():.2f} s, maior gap {df.gap.max():.1e}")


def referencias(t):
    """ibov e cdi na mesma janela [t - 3 anos, t) do mu e sigma"""
    brutos = dados.carregar()
    ibov = dados.retornos_simples(dados.janela(brutos.ibov, t)).dropna()
    cdi = dados.janela(dados.carregar_cdi(), t)
    a = dados.PREGOES_ANO
    return pd.DataFrame([
        {"referencia": "Ibovespa", "retorno": ibov.mean() * a, "vol": ibov.std() * np.sqrt(a),
         "inicio": ibov.index.min().date(), "fim": ibov.index.max().date()},
        {"referencia": "CDI", "retorno": cdi.mean() * a, "vol": cdi.std() * np.sqrt(a),
         "inicio": cdi.index.min().date(), "fim": cdi.index.max().date()},
    ])


def grafico(res, ref, destino, t, sigma):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # paleta de cores
    cores = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    tinta, tinta2, muda, grade = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": tinta2,
                         "xtick.color": muda, "ytick.color": muda})

    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    for ax, zoom in zip(eixos, [False, True]):
        livre = res[res.serie == "sem_card"]
        ax.plot(livre.vol * 100, livre.retorno * 100, color=tinta2, lw=1.5, ls="--",
                label="Sem cardinalidade", zorder=5)
        for cor, K in zip(cores, KS):
            d = res[res.serie == f"K{K}"]
            ax.plot(d.vol * 100, d.retorno * 100, color=cor, lw=2, label=f"K = {K}", zorder=3)
        if not zoom:
            for _, r in ref.iterrows():
                ax.plot(r.vol * 100, r.retorno * 100, "o", ms=8, color=tinta, mec="white", mew=2,
                        zorder=4)
                ax.annotate(r.referencia, (r.vol * 100, r.retorno * 100), xytext=(7, -3),
                            textcoords="offset points", fontsize=9, color=tinta)
        ax.grid(color=grade, lw=0.6)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        ax.set_xlabel("Volatilidade anualizada (%)")
        ax.set_ylabel("Retorno esperado anualizado (%)")
    # zoom na parte de menor risco, onde as fronteiras se separam
    v0 = res.vol.min() * 100
    eixos[1].set_xlim(v0 - 0.3, v0 + 4)
    sub = res[res.vol * 100 <= v0 + 4]
    eixos[1].set_ylim(sub.retorno.min() * 100 - 1, sub.retorno.max() * 100 + 1)
    eixos[0].set_title("(a) Fronteiras e referências", fontsize=10, color=tinta, loc="left")
    eixos[1].set_title("(b) Detalhe da região de menor risco", fontsize=10, color=tinta, loc="left")
    eixos[0].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.30, 0.42))
    fig.suptitle(f"Fronteira eficiente por cardinalidade — decisão em {t}, Σ {sigma}, "
                 f"l = {L:.0%}, u = {U:.0%}", fontsize=10, color=tinta, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(destino / f"fronteira.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", default="2014-01-02")
    ap.add_argument("--sigma", choices=["amostral", "lw"], default="amostral")
    ap.add_argument("--sem-b3", action="store_true", help="só Yahoo (sem as séries do COTAHIST)")
    args = ap.parse_args()
    tag = f"{args.t}_{'semb3' if args.sem_b3 else 'b3'}"
    if not EXE.exists():
        sys.exit("Compile antes: cmake -S . -B build && cmake --build build")

    inst = RAIZ / "data" / "instancias" / tag / args.sigma
    if not inst.exists():
        sys.exit(f"Instância ausente: rode  python3 scripts/preparar_dados.py --t {args.t}"
                 + (" --sem-b3" if args.sem_b3 else ""))
    mu = pd.read_csv(inst / "assets.csv", header=None, index_col=0)[1].values
    N = len(mu)
    destino = RAIZ / "results" / "fronteira" / f"{tag}_{args.sigma}"
    destino.mkdir(parents=True, exist_ok=True)

    linhas, pesos = [], []
    for K in KS:
        tracar(inst, mu, f"K{K}", K, L, U, linhas, pesos)
    # sem cardinalidade: K = N e l = 0 (todo z_i = 1, w_i livre em [0, u])
    tracar(inst, mu, "sem_card", N, 0.0, U, linhas, pesos)

    res = pd.DataFrame(linhas)
    ref = referencias(pd.Timestamp(args.t))
    cdi_mu = float(ref.loc[ref.referencia == "CDI", "retorno"].iloc[0])
    res["sharpe_cdi"] = (res.retorno - cdi_mu) / res.vol
    livre0 = res[res.serie == "sem_card"].iloc[0]
    ref = pd.concat([ref, pd.DataFrame([{"referencia": "Var. mínima sem cardinalidade",
                                         "retorno": livre0.retorno, "vol": livre0.vol,
                                         "inicio": None, "fim": None}])], ignore_index=True)
    ref["sharpe_cdi"] = (ref.retorno - cdi_mu) / ref.vol

    res.to_csv(destino / "resolucoes.csv", index=False)
    pd.DataFrame(pesos).to_csv(destino / "pesos.csv", index=False)
    ref.to_csv(destino / "referencias.csv", index=False)
    grafico(res, ref.iloc[:2], destino, args.t, args.sigma)

    print(f"\nN = {N}; {len(res)} resoluções, {res.otimo.sum()} ótimas, "
          f"{(~res.viavel).sum()} inviáveis; tempo total {res.tempo_s.sum():.1f} s")
    print(ref.to_string(index=False))
    print(f"Saídas em {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
