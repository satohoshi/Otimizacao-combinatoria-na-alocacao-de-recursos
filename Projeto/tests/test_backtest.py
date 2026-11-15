"""testes da decisao e da simulacao do backtest

uso:  python3 tests/test_backtest.py   (precisa do build/portfolio_opt compilado)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest  # noqa: E402
import dados  # noqa: E402
from test_dados import brutos, corromper_futuro  # noqa: E402

DATAS_T = ["2016-07-01", "2020-04-01"]
KS = {3: (0.01, 0.40), 9: (0.01, 0.30)}


def corromper_cdi(cdi, t, seed=0):
    rng = np.random.default_rng(seed)
    cdi = cdi.copy()
    fut = cdi.index >= pd.Timestamp(t)
    cdi[fut] = rng.uniform(0, 0.01, fut.sum())
    return cdi


def test_decisao_sem_vazamento_do_futuro():
    """precos, volumes, ibov e cdi a partir de t viram lixo, a carteira escolhida
    em t tem q ser a mesma"""
    cdi = dados.carregar_cdi()
    for t in DATAS_T:
        _, a = backtest.decidir(brutos(), cdi, t, ks=KS)
        _, c = backtest.decidir(corromper_futuro(brutos(), t), corromper_cdi(cdi, t), t, ks=KS)
        a, c = pd.DataFrame(a), pd.DataFrame(c)
        assert len(a) > 0, t
        pd.testing.assert_frame_equal(a, c)


def test_decisao_corrupcao_e_detectada():
    """controle: lixo comecando 30 dias antes de t tem q mudar a decisao"""
    cdi = dados.carregar_cdi()
    t = pd.Timestamp(DATAS_T[0])
    t_antes = t - pd.Timedelta(days=30)
    _, a = backtest.decidir(brutos(), cdi, t, ks=KS)
    _, c = backtest.decidir(corromper_futuro(brutos(), t_antes), corromper_cdi(cdi, t_antes), t, ks=KS)
    a, c = pd.DataFrame(a), pd.DataFrame(c)
    assert not (a.shape == c.shape and np.allclose(a.peso, c.peso) and (a.ticker == c.ticker).all())


def test_decisao_nsga2_cvar_sem_vazamento_e_reprodutivel():
    """mesmo teste de vazamento com cvar + nsga2 e mesma semente: lixo a partir
    de t n muda a decisao em t"""
    cdi = dados.carregar_cdi()
    t = DATAS_T[0]
    _, a = backtest.decidir(brutos(), cdi, t, ks=KS, medida="cvar", metodo="nsga2")
    _, a2 = backtest.decidir(brutos(), cdi, t, ks=KS, medida="cvar", metodo="nsga2")
    _, c = backtest.decidir(corromper_futuro(brutos(), t), corromper_cdi(cdi, t), t, ks=KS,
                            medida="cvar", metodo="nsga2")
    a, a2, c = pd.DataFrame(a), pd.DataFrame(a2), pd.DataFrame(c)
    assert len(a) > 0
    pd.testing.assert_frame_equal(a, a2)
    pd.testing.assert_frame_equal(a, c)


def test_simulacao_deriva_e_custo():
    """dois ativos 50/50 e A dobra no dia 2: pesos vao pra 2/3 e 1/3 e o rebal
    no dia 3 negocia 1/6 do valor em cada"""
    idx = pd.bdate_range("2014-01-02", periods=4)
    ret = pd.DataFrame({"A": [0, 1.0, 0, 0], "B": [0, 0, 0, 0]}, idx, dtype=float)
    w = pd.Series({"A": 0.5, "B": 0.5})
    v, to, custos = backtest.simular(ret, {idx[0]: w, idx[2]: w}, custo=0.01)
    v0 = backtest.V0 * (1 - 0.01)                     # compra inicial paga 1%
    assert np.isclose(v.iloc[1], 1.5 * v0)
    assert np.isclose(to[0], (1.5 * v0 / 6 * 2) / (2 * 1.5 * v0))   # 1/6 do valor
    assert np.isclose(v.iloc[2], 1.5 * v0 * (1 - 0.01 / 3))
    assert np.isclose(custos, backtest.V0 * 0.01 + 1.5 * v0 * 0.01 / 3)


if __name__ == "__main__":
    for nome, f in list(globals().items()):
        if nome.startswith("test_"):
            f()
            print("ok", nome)
