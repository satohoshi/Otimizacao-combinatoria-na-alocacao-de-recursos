"""valida o modelo c++/cplex com enumeracao completa numa instancia pequena

com N = 10 (os 10 primeiros em ordem alfabetica da instancia de 2014) testa todos
os C(N, K) subconjuntos, resolve o qp continuo
    min w' S w   s.a.  soma w = 1,  mu' w >= R,  l <= w <= u
com scipy (SLSQP, sem cplex) e pega o melhor. compara o risco e a viabilidade
com a saida do build/portfolio_opt

uso:
    python3 scripts/validar_enumeracao.py
saida:
    results/validacao_enumeracao.csv
"""
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

RAIZ = Path(__file__).resolve().parents[1]
EXE = RAIZ / "build" / "portfolio_opt"
ORIGEM = RAIZ / "data" / "instancias" / "2014-01-02" / "amostral"
DESTINO = RAIZ / "data" / "instancias" / "validacao_N10"
N = 10
TOL_REL = 1e-5


def montar_instancia():
    mu = pd.read_csv(ORIGEM / "assets.csv", header=None, index_col=0)[1]
    sig = np.loadtxt(ORIGEM / "covariance.csv", delimiter=",")
    idx = np.argsort(mu.index)[:N]
    mu, sig = mu.iloc[idx], sig[np.ix_(idx, idx)]
    DESTINO.mkdir(parents=True, exist_ok=True)
    mu.to_csv(DESTINO / "assets.csv", header=False, float_format="%.10f")
    np.savetxt(DESTINO / "covariance.csv", sig, delimiter=",", fmt="%.12e")
    return mu.values, sig, list(mu.index)


def qp_subconjunto(mu, sig, R, l, u):
    """menor variancia no subconjunto (None se for inviavel)"""
    k = len(mu)
    if k * l > 1 + 1e-12 or k * u < 1 - 1e-12:
        return None
    if R is not None:
        # retorno max viavel no subconjunto, pra saber se eh viavel
        lp = linprog(-mu, A_eq=np.ones((1, k)), b_eq=[1], bounds=[(l, u)] * k, method="highs")
        if -lp.fun < R - 1e-10:
            return None
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(k)}]
    if R is not None:
        cons.append({"type": "ineq", "fun": lambda w: mu @ w - R, "jac": lambda w: mu})
    melhor = None
    inicios = [np.full(k, 1 / k)]
    if R is not None:
        inicios.append(np.clip(lp.x, l, u))
    for w0 in inicios:
        res = minimize(lambda w: w @ sig @ w, w0, jac=lambda w: 2 * sig @ w, method="SLSQP",
                       bounds=[(l, u)] * k, constraints=cons, options={"ftol": 1e-15, "maxiter": 1000})
        w = res.x
        ok = abs(w.sum() - 1) < 1e-8 and (R is None or mu @ w >= R - 1e-8) \
            and (w >= l - 1e-9).all() and (w <= u + 1e-9).all()
        if ok and (melhor is None or res.fun < melhor):
            melhor = float(res.fun)
    return melhor


def enumerar(mu, sig, K, R, l, u):
    melhor, sub = None, None
    for S in itertools.combinations(range(len(mu)), K):
        S = list(S)
        v = qp_subconjunto(mu[S], sig[np.ix_(S, S)], R, l, u)
        if v is not None and (melhor is None or v < melhor):
            melhor, sub = v, S
    return melhor, sub


def cplex(K, R, l, u):
    cmd = [str(EXE), "--assets", str(DESTINO / "assets.csv"), "--cov", str(DESTINO / "covariance.csv"),
           "--K", str(K), "--l", str(l), "--u", str(u), "--gap", "1e-9", "--time-limit", "60"]
    if R is not None:
        cmd += ["--R", repr(R)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    if not EXE.exists():
        sys.exit("Compile antes: cmake -S . -B build && cmake --build build")
    mu, sig, nomes = montar_instancia()
    print("Instância N=10:", nomes)
    linhas = []
    for (l, u), K in itertools.product([(0.0, 1.0), (0.01, 0.30), (0.05, 0.50)], [1, 2, 3, 4, 5, 7]):
        if K * u < 1 - 1e-12:      # inviavel, testa so uma vez com R=None
            casos = [None]
        else:
            vmin, _ = enumerar(mu, sig, K, None, l, u)
            # retorno max com K ativos
            rmax = max(-linprog(-mu[list(S)], A_eq=np.ones((1, K)), b_eq=[1], bounds=[(l, u)] * K,
                                method="highs").fun
                       for S in itertools.combinations(range(N), K) if K * l <= 1 + 1e-12)
            r0 = float(mu.min())
            casos = [None] + [r0 + f * (rmax - r0) for f in (0.5, 0.9, 1.0)] + [rmax + 0.01]
        for R in casos:
            ref, sub = enumerar(mu, sig, K, R, l, u)
            c = cplex(K, R, l, u)
            escolhidos = sorted(c["pesos"])
            ok_viab = (ref is not None) == c["viavel"]
            dif = abs(c["risco"] - ref) / max(ref, 1e-12) if ref is not None and c["viavel"] else 0.0
            ok = ok_viab and dif <= TOL_REL
            linhas.append({"K": K, "l": l, "u": u, "R": R, "enum_risco": ref,
                           "enum_ativos": ",".join(nomes[i] for i in sub) if sub else "",
                           "cplex_status": c["status"], "cplex_risco": c["risco"] if c["viavel"] else None,
                           "cplex_ativos": ",".join(escolhidos), "dif_rel": dif, "ok": ok})
            print(f"K={K} l={l} u={u} R={'—' if R is None else f'{R:.4f}'}: "
                  f"enum={ref if ref is None else f'{ref:.8f}'} cplex={c['status']} "
                  f"{c['risco']:.8f} dif={dif:.1e} {'OK' if ok else 'FALHA'}")
    df = pd.DataFrame(linhas)
    (RAIZ / "results").mkdir(exist_ok=True)
    df.to_csv(RAIZ / "results" / "validacao_enumeracao.csv", index=False)
    print(f"\n{df.ok.sum()}/{len(df)} casos OK; maior diferença relativa = {df.dif_rel.max():.2e}")
    sys.exit(0 if df.ok.all() else 1)


if __name__ == "__main__":
    main()
