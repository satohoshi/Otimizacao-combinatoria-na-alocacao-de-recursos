"""testes do tratamento de dados

uso:  python3 -m pytest tests/test_dados.py   (ou python3 tests/test_dados.py)
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dados  # noqa: E402

DATAS_T = ["2014-01-02", "2016-07-01", "2020-03-02"]
_BRUTOS = None


def brutos():
    global _BRUTOS
    if _BRUTOS is None:
        _BRUTOS = dados.carregar()
    return _BRUTOS


def corromper_futuro(b, t, seed=0):
    """troca por lixo tudo a partir de t (inclusive), se a estimacao em t mudar
    algum dado >= t entrou na conta"""
    rng = np.random.default_rng(seed)
    t = pd.Timestamp(t)

    def lixo(df):
        df = df.copy()
        fut = df.index >= t
        df.loc[fut] = rng.uniform(0.01, 1e4, size=df.loc[fut].shape)
        return df

    ib = b.ibov.copy()
    ib[ib.index >= t] = rng.uniform(1, 1e6, size=(ib.index >= t).sum())
    return replace(b, precos=lixo(b.precos), fechamento=lixo(b.fechamento), volume=lixo(b.volume), ibov=ib,
                   negocios=lixo(b.negocios))


def test_sem_vazamento_do_futuro():
    for t in DATAS_T:
        a = dados.estimar(brutos(), t)
        c = dados.estimar(corromper_futuro(brutos(), t), t)
        assert a.tickers == c.tickers, t
        pd.testing.assert_series_equal(a.mu, c.mu)
        pd.testing.assert_frame_equal(a.sigma, c.sigma)
        pd.testing.assert_frame_equal(a.sigma_lw, c.sigma_lw)
        assert a.retornos.index.max() < pd.Timestamp(t)
        assert a.fim < pd.Timestamp(t)


def test_corrupcao_e_detectada():
    """controle: se o lixo comecar antes de t a estimacao tem q mudar
    (pra garantir q o teste de cima n passa por sorte)"""
    t = pd.Timestamp("2014-01-02")
    a = dados.estimar(brutos(), t)
    c = dados.estimar(corromper_futuro(brutos(), t - pd.Timedelta(days=30)), t)
    assert not np.allclose(a.mu.reindex(c.mu.index).fillna(0), c.mu)


def test_sigma_psd():
    for t in DATAS_T:
        e = dados.estimar(brutos(), t)
        assert dados.eh_psd(e.sigma)[0] and dados.eh_psd(e.sigma_lw)[0], t
        assert np.allclose(e.sigma.values, e.sigma.values.T)


def test_degrau_neutralizado_e_real_mantido():
    idx = pd.bdate_range("2020-01-01", periods=6)
    ret = pd.DataFrame({"A": [0.01, -0.66, 0.0, 0.01, 0.0, 0.0],      # degrau sem mercado
                        "B": [0.0, -0.30, 0.01, 0.0, 0.0, 0.0]}, idx)  # queda forte mas abaixo de 50%
    ibov = pd.Series(0.0, idx)
    limpo, ev = dados.limpar_extremos(ret, ibov)
    assert limpo.at[idx[1], "A"] == 0.0
    assert limpo.at[idx[1], "B"] == -0.30          # |r| < 50%, mantem
    assert set(ev.classificacao) == {"degrau", "real"}


if __name__ == "__main__":
    for nome, f in list(globals().items()):
        if nome.startswith("test_"):
            f()
            print("ok", nome)
