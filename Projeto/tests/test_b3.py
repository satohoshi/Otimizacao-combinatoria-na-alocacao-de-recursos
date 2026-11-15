"""testes das series da b3 (cotahist) e da regra de deslistagem

uso:  python3 -m pytest tests/test_b3.py   (ou python3 tests/test_b3.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ajustar_b3  # noqa: E402
import backtest  # noqa: E402
import dados  # noqa: E402
import parse_cotahist  # noqa: E402
from test_dados import brutos, corromper_futuro  # noqa: E402

RAIZ = dados.RAIZ


def registro_sintetico():
    """registro tipo 01 montado campo a campo pelo layout (245 colunas)"""
    campos = [
        "01", "20131230", "02", "TEST3".ljust(12), "010", "TESTE SA".ljust(12), "ON      NM",
        "   ", "R$  ",
        "0000002450000",       # PREABE  24500.00 por lote
        "0000002600000",       # PREMAX
        "0000002400000",       # PREMIN
        "0000002500000",       # PREMED
        "0000002550000",       # PREULT  25500.00 por lote -> 25.50 por acao
        "0000000000000", "0000000000000",          # PREOFC, PREOFV
        "00123",               # TOTNEG
        "000000000000045000",  # QUATOT
        "000000000114750000",  # VOLTOT  1147500.00
        "0000000000000", "0", "99991231",          # PREEXE, INDOPC, DATVEN
        "0001000",             # FATCOT: lote de mil acoes
        "0000000000000",       # PTOEXE
        "BRTESTACNOR1", "101",
    ]
    linha = "".join(campos)
    assert len(linha) == 245
    return linha


def test_parser_registro_sintetico():
    r = parse_cotahist.ler_registro(registro_sintetico())
    assert r["tipreg"] == "01" and r["codbdi"] == "02" and r["tpmerc"] == "010"
    assert r["data"] == pd.Timestamp("2013-12-30")
    assert r["codneg"] == "TEST3" and r["nomres"] == "TESTE SA" and r["codisi"] == "BRTESTACNOR1"
    assert r["fatcot"] == 1000
    assert np.isclose(r["preult"], 25.50) and np.isclose(r["preabe"], 24.50)    # /100 e /FATCOT
    assert np.isclose(r["premax"], 26.00) and np.isclose(r["premin"], 24.00)
    assert np.isclose(r["voltot"], 1_147_500.00)                                # /100, sem FATCOT
    assert r["totneg"] == 123 and r["quatot"] == 45_000 and r["dismes"] == 101


def test_desdobramento_2_por_1_sem_degrau():
    """preco cai de 20 pra 10.20 no dia ex: o ajuste aplica 1/2 nos precos de antes
    e o retorno do dia vira +2%"""
    idx = pd.bdate_range("2015-01-05", periods=10)
    preco = [20.0, 20.1, 19.9, 20.0, 20.0, 10.2, 10.3, 10.2, 10.25, 10.3]
    esp = ["ON      NM"] * 5 + ["ON  EX  NM"] * 3 + ["ON      NM"] * 2
    s = pd.DataFrame({"fechamento": preco, "especi": esp, "negocios": 1000, "ativo": "TEST3",
                      "ticker": "TEST3"}, idx)
    cand = pd.DataFrame(ajustar_b3.candidatos(s, pd.Series(0.0, idx)))
    assert len(cand) == 1 and cand.data.iloc[0] == idx[5]
    classe, fator, aplicado, _ = ajustar_b3.classificar(cand.iloc[0])
    assert classe == "evento em ações" and aplicado and np.isclose(fator, 0.5)
    adj = s.fechamento * np.where(idx < idx[5], fator, 1.0)
    ret = adj.pct_change().dropna()
    assert ret.abs().max() < 0.03
    assert np.isclose(ret.iloc[4], 10.2 / 10.0 - 1)


def test_sem_b3_reproduz_estimacao_antiga():
    """com usar_b3=False o mu, sigma e sigma_lw de 2014 tem q ser iguais aos ja
    gravados em data/processado/2014-01-02/"""
    e = dados.estimar(dados.carregar(usar_b3=False), "2014-01-02")
    pasta = RAIZ / "data" / "processado" / "2014-01-02"
    ler = dict(index_col=0, float_precision="round_trip")
    mu = pd.read_csv(pasta / "mu.csv", **ler).mu_anual
    assert list(e.mu.index) == list(mu.index)
    assert (e.mu.values == mu.values).all()
    assert (e.sigma.values == pd.read_csv(pasta / "sigma_amostral.csv", **ler).values).all()
    assert (e.sigma_lw.values == pd.read_csv(pasta / "sigma_lw.csv", **ler).values).all()


def test_sem_vazamento_com_series_b3():
    """com a b3 na base, lixo em precos, volumes, negocios e ibov a partir de t
    n muda a estimacao em t, e as series da b3 entram nela"""
    b = brutos()
    for t in ["2014-01-02", "2017-01-02", "2020-01-02"]:
        a = dados.estimar(b, t)
        c = dados.estimar(corromper_futuro(b, t), t)
        assert any(b.origem[tk] == "b3" for tk in a.tickers), t
        assert a.tickers == c.tickers, t
        pd.testing.assert_series_equal(a.mu, c.mu)
        pd.testing.assert_frame_equal(a.sigma, c.sigma)
        pd.testing.assert_frame_equal(a.eventos, c.eventos)


def test_decisao_sem_vazamento_com_series_b3():
    cdi = dados.carregar_cdi()
    t = "2014-01-02"
    ks = {9: (0.01, 0.30)}
    _, a = backtest.decidir(brutos(), cdi, t, ks=ks)
    rng = np.random.default_rng(1)
    cdi_c = cdi.copy()
    cdi_c[cdi_c.index >= t] = rng.uniform(0, 0.01, (cdi_c.index >= t).sum())
    _, c = backtest.decidir(corromper_futuro(brutos(), t), cdi_c, t, ks=ks)
    pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(c))


def test_colapso_real_nao_vira_degrau():
    """MMXM3 caiu 52.8% em 14/10/2020 com 9232 negocios (mediana ~700), entao eh
    negociacao real. com os negocios a regra mantem o retorno, sem eles (como no
    yahoo) o mesmo dia seria zerado como degrau"""
    b = brutos()
    p = dados.preencher(dados.janela(b.precos[["MMXM3"]], "2021-01-04"))
    ret = dados.retornos_simples(p)
    ret_ibov = dados.retornos_simples(dados.janela(b.ibov, "2021-01-04"))
    neg = dados.janela(b.negocios[["MMXM3"]], "2021-01-04")
    d = pd.Timestamp("2020-10-14")
    assert ret.at[d, "MMXM3"] < -0.5

    limpo, ev = dados.limpar_extremos(ret, ret_ibov, neg)
    assert limpo.at[d, "MMXM3"] == ret.at[d, "MMXM3"]
    assert ev.set_index("data").classificacao[d.date()] == "real"
    assert "degrau" not in set(ev.classificacao)

    limpo_sem, ev_sem = dados.limpar_extremos(ret, ret_ibov)
    assert limpo_sem.at[d, "MMXM3"] == 0.0
    assert ev_sem.set_index("data").classificacao[d.date()] == "degrau"


def test_colapso_ogx_mantido_na_estimacao():
    """os retornos da OGX em 2013 (caiu ~97% no ano) entram na janela de 2014
    sem nenhum dia zerado"""
    e = dados.estimar(brutos(), "2014-01-02")
    assert "OGXP3" in e.tickers
    ev = e.eventos[e.eventos.ticker == "OGXP3"]
    assert len(ev) > 0 and "degrau" not in set(ev.classificacao)
    acum = (1 + e.retornos.OGXP3.loc["2013-01-01":]).prod() - 1
    assert acum < -0.9


def test_deslistagem_liquida_e_caixa_rende_cdi():
    """A e B 50/50 em d0 (100 mil). A sobe 10% em d1, cai 20% em d2 e para de ser
    negociado, B sobe 5% em d2 e d3. cdi de 1% ao dia
      d1: A = 55.000, B = 50.000                     -> 105.000
      d2: A = 44.000 (liquidado), B = 52.500         -> 96.500
      d3: caixa = 44.440, B = 55.125                 -> 99.565
      d4: caixa = 44.884,40, B = 55.125              -> 100.009,40
    com a regra antiga (congelar) d3 = d4 = 44.000 + 55.125 = 99.125"""
    idx = pd.bdate_range("2014-01-02", periods=5)
    ret = pd.DataFrame({"A": [0, 0.10, -0.20, 0, 0], "B": [0, 0, 0.05, 0.05, 0]}, idx, dtype=float)
    w = {idx[0]: pd.Series({"A": 0.5, "B": 0.5})}
    cdi = pd.Series(0.01, idx)
    reg = []
    v, to, custos = backtest.simular(ret, w, 0.0, fim={"A": idx[2]}, caixa=cdi, registro=reg)
    assert np.allclose(v.values, [100_000, 105_000, 96_500, 99_565, 100_009.40])
    assert len(reg) == 1 and reg[0]["data"] == idx[2] and reg[0]["ticker"] == "A"
    assert np.isclose(reg[0]["valor"], 44_000)
    v_old, _, _ = backtest.simular(ret, w, 0.0)
    assert np.allclose(v_old.values, [100_000, 105_000, 96_500, 99_125, 99_125])

    # rebal em d3 reinveste o caixa, 100% em B
    w2 = {**w, idx[3]: pd.Series({"B": 1.0})}
    v2, to2, _ = backtest.simular(ret, w2, 0.0, fim={"A": idx[2]}, caixa=cdi)
    assert np.allclose(v2.values, [100_000, 105_000, 96_500, 99_565, 99_565])
    assert np.isclose(to2[0], 44_440 / (2 * 99_565))


def test_ultimos_pregoes_b3():
    """ativos da b3 q acabam antes do fim dos dados sao deslistados, IMCH3
    (junto com MEAL3) continua negociando e n eh"""
    b = brutos()
    fim = backtest.ultimos_pregoes(b)
    b3 = [tk for tk, o in b.origem.items() if o == "b3"]
    assert set(b3) - {"IMCH3"} <= set(fim) and "IMCH3" not in fim
    assert fim["OGXP3"] == pd.Timestamp("2019-01-10") and fim["AEDU3"] == pd.Timestamp("2014-07-03")


def test_base_consolidada_igual_ao_pipeline():
    """a matriz de dados_brutos/consolidado/ eh igual a q o dados.carregar monta
    (depois dos TRECHOS_INVALIDOS) e o inventario tem os 100 tickers"""
    pasta = RAIZ / "dados_brutos" / "consolidado"
    m = pd.read_csv(pasta / "matriz_adj_close.csv", index_col=0, parse_dates=True, float_precision="round_trip")
    for tk, inicio in dados.TRECHOS_INVALIDOS.items():
        m.loc[m.index < inicio, tk] = np.nan
    b = brutos()
    assert list(m.columns) == list(b.precos.columns) and (m.index == b.precos.index).all()
    assert np.array_equal(m.values, b.precos.values, equal_nan=True)
    inv = pd.read_csv(pasta / "inventario_ativos.csv")
    assert len(inv) == 100 and inv.origem.value_counts().to_dict() == {"yahoo": 73, "b3": 26, "sem dados": 1}


if __name__ == "__main__":
    for nome, f in list(globals().items()):
        if nome.startswith("test_"):
            f()
            print("ok", nome)
