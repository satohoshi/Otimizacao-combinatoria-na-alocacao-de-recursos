"""testes do cvar (cplex) e do nsga2

uso:  python3 tests/test_metodos.py   (ou python3 -m pytest tests/test_metodos.py)

usa uma instancia sintetica pequena (N = 8, S = 120) numa pasta temporaria pra
enumeracao completa ser rapida
"""
import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

RAIZ = Path(__file__).resolve().parents[1]
EXE = RAIZ / "build" / "portfolio_opt"
N, S, BETA = 8, 120, 0.95
L, U = 0.05, 0.6
_INST = None


def instancia():
    """retornos diarios sinteticos com correlacao e cauda pesada (t de student)"""
    global _INST
    if _INST is None:
        rng = np.random.default_rng(42)
        fator = rng.standard_t(4, size=(S, 1)) * 0.012
        r = 0.0004 + rng.uniform(-3e-4, 1e-3, N) + fator * rng.uniform(0.5, 1.5, N) \
            + rng.standard_t(5, size=(S, N)) * rng.uniform(0.006, 0.02, N)
        pasta = Path(tempfile.mkdtemp(prefix="inst_metodos_"))
        with open(pasta / "assets.csv", "w") as f:
            for i, m in enumerate(r.mean(0) * 252):
                f.write(f"A{i},{m:.10f}\n")
        np.savetxt(pasta / "covariance.csv", np.cov(r.T) * 252, delimiter=",", fmt="%.12e")
        np.savetxt(pasta / "scenarios.csv", r, delimiter=",", fmt="%.12e")
        _INST = (pasta, r)
    return _INST


def rodar(*extra):
    pasta, _ = instancia()
    cmd = [str(EXE), "--assets", str(pasta / "assets.csv"), "--cov", str(pasta / "covariance.csv"),
           "--cenarios", str(pasta / "scenarios.csv"), "--beta", str(BETA), "--l", str(L), "--u", str(U),
           "--threads", "1", *map(str, extra)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def vetor(pesos):
    w = np.zeros(N)
    for tk, p in pesos.items():
        w[int(tk[1:])] = p
    return w


def cvar_numpy(r, w, beta=BETA):
    """cvar pela definicao: media das perdas na cauda, com o peso fracionario do var"""
    perda = np.sort(-r @ w)[::-1]
    m = (1 - beta) * len(perda)        # qtd (fracionaria) de cenarios na cauda
    k = int(np.floor(m + 1e-9))
    return (perda[:k].sum() + (m - k) * perda[k]) / m if k < len(perda) else perda.mean()


def cvar_lp(r, sub, R=None):
    """cvar minimo com o subconjunto `sub` fixo (lp do rockafellar e uryasev, scipy/highs)"""
    k, s = len(sub), len(r)
    mu = r.mean(0) * 252
    # variaveis: w (k), alfa, u (s)
    c = np.r_[np.zeros(k), 1.0, np.full(s, 1 / ((1 - BETA) * s))]
    A = np.hstack([-r[:, sub], -np.ones((s, 1)), -np.eye(s)])        # -r w - alfa - u <= 0
    b = np.zeros(s)
    if R is not None:
        A = np.vstack([A, np.r_[-mu[sub], 0.0, np.zeros(s)]])
        b = np.r_[b, -R]
    res = linprog(c, A_ub=A, b_ub=b, A_eq=[np.r_[np.ones(k), 0.0, np.zeros(s)]], b_eq=[1.0],
                  bounds=[(L, U)] * k + [(None, None)] + [(0, None)] * s, method="highs")
    return res.fun if res.status == 0 else np.inf


def test_cvar_avaliacao_confere_com_definicao():
    _, r = instancia()
    for K in (2, 4):
        sol = rodar("--K", K, "--risco", "cvar")
        w = vetor(sol["pesos"])
        assert abs(sol["cvar"] - cvar_numpy(r, w)) < 1e-9, (sol["cvar"], cvar_numpy(r, w))
        assert abs(sol["risco"] - sol["cvar"]) < 1e-9          # fo do cplex = avaliacao
        assert sol["var"] <= sol["cvar"] + 1e-12               # VaR <= CVaR
        assert abs(sol["variancia"] - w @ np.loadtxt(instancia()[0] / "covariance.csv", delimiter=",") @ w) < 1e-9


def test_cvar_cplex_igual_a_enumeracao():
    _, r = instancia()
    mu = r.mean(0) * 252
    for K in (2, 3, 5):
        for frac in (None, 0.5, 0.9):
            R = None
            if frac is not None:
                top = np.sort(mu)[::-1][:K]                    # retorno max com K ativos em [L, U]
                w = np.full(K, L)
                resto = 1 - K * L
                for i in range(K):
                    a = min(U - L, resto)
                    w[i] += a
                    resto -= a
                R = float(mu.min() + frac * (top @ w - mu.min()))
            enum = min(cvar_lp(r, list(sub), R) for sub in itertools.combinations(range(N), K))
            sol = rodar("--K", K, "--risco", "cvar", *(["--R", repr(R)] if R is not None else []))
            assert sol["viavel"] == np.isfinite(enum), (K, R)
            if sol["viavel"]:
                assert abs(sol["risco"] - enum) <= 1e-4 * abs(enum) + 1e-9, (K, R, sol["risco"], enum)


def test_nsga2_frente_viavel_e_nao_supera_otimo():
    for medida in ("variancia", "cvar"):
        for K in (2, 4):
            ga = rodar("--K", K, "--risco", medida, "--metodo", "nsga2", "--pop", 40, "--geracoes", 80,
                       "--seed", 7)
            assert ga["viavel"] and len(ga["frente"]) > 1
            opt = rodar("--K", K, "--risco", medida)
            ret_ant = -np.inf
            for s in ga["frente"]:
                w = np.array(list(s["pesos"].values()))
                assert len(w) == K and abs(w.sum() - 1) < 1e-9
                assert w.min() >= L - 1e-12 and w.max() <= U + 1e-12
                assert s["risco"] >= opt["risco"] * (1 - 1e-4) - 1e-12   # nunca abaixo do otimo
                assert s["retorno"] > ret_ant                              # frente nao dominada
                ret_ant = s["retorno"]


def test_nsga2_deterministico_pela_semente():
    a = rodar("--K", 3, "--metodo", "nsga2", "--pop", 20, "--geracoes", 30, "--seed", 3)
    b = rodar("--K", 3, "--metodo", "nsga2", "--pop", 20, "--geracoes", 30, "--seed", 3)
    c = rodar("--K", 3, "--metodo", "nsga2", "--pop", 20, "--geracoes", 30, "--seed", 4)
    assert a["frente"] == b["frente"]
    assert a["frente"] != c["frente"]


def test_nsga2_inviavel_por_construcao():
    sol = rodar("--K", 1, "--metodo", "nsga2")              # 1 x 0.6 < 1
    assert not sol["viavel"] and sol["frente"] == []


if __name__ == "__main__":
    for nome, f in list(globals().items()):
        if nome.startswith("test_"):
            f()
            print(f"{nome}: OK")
