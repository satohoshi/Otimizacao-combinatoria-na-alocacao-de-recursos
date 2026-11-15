"""tratamento dos dados e estimacao de mu e sigma

as estimacoes recebem uma data t e so usam pregoes antes de t, a limpeza
tambem eh feita dentro da janela ja cortada pra n olhar o futuro

fontes: yahoo (preco ajustado) e, pras acoes q n tem no yahoo, o cotahist da b3
ajustado (scripts/ajustar_b3.py). se tiver nas duas vale o yahoo
carregar(usar_b3=False) usa so o yahoo
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

RAIZ = Path(__file__).resolve().parents[1]
BRUTOS = RAIZ / "dados_brutos" / "yahoo"
BRUTOS_B3 = RAIZ / "dados_brutos" / "b3"

PREGOES_ANO = 252
JANELA_ANOS = 3             # janela de estimacao
LIMITE_FFILL = 5            # max de pregoes seguidos preenchidos com o ultimo preco
LIMITE_RET_ZERO = 0.20      # max de retornos zerados (preco parado) na janela
LIMIAR_EXTREMO = 0.20       # |r| acima disso eh investigado
LIMIAR_REVERSAO = 0.10      # retorno oposto no dia seguinte = reversao
LIMIAR_DEGRAU = 0.50        # |r| acima disso sem reversao = evento n ajustado
LIMIAR_MERCADO = 0.05       # |r_ibov| abaixo disso = mercado n explica
LIMIAR_CORRELACAO = 0.90
LIMIAR_NEGOCIOS = 0.5       # negocios >= 50% da mediana de 60 pregoes = negociacao real

# trechos q olhei na mao e sao invalidos (viram preco faltante)
# ticker -> primeira data valida
# TIMP3: vol de 113% em jan-jul/2011 contra 34% depois e salto de +24% em
#        01/08/2011 sem o mercado mexer, provavelmente troca de classe no yahoo
TRECHOS_INVALIDOS = {"TIMP3": "2011-08-01"}


# carga
@dataclass
class Brutos:
    precos: pd.DataFrame        # preco ajustado (data x ticker), no calendario do ibov
    fechamento: pd.DataFrame    # fechamento sem ajuste, pro volume financeiro
    volume: pd.DataFrame        # volume em qtd de acoes
    ibov: pd.Series
    negocios: pd.DataFrame = field(default_factory=pd.DataFrame)   # qtd de negocios (so b3)
    origem: dict = field(default_factory=dict)                     # ticker -> yahoo ou b3


def carregar(pasta=BRUTOS, usar_b3=True, pasta_b3=BRUTOS_B3):
    """le os csv da coleta e alinha tudo no calendario do ibov

    dias q o yahoo tem acao mas n tem ^BVSP (feriado da b3) sao descartados,
    dias do ibov sem preco ficam NaN e sao tratados no preencher
    com usar_b3 o cotahist entra so pros tickers q n tem no yahoo
    """
    longo = pd.read_csv(pasta / "precos_yahoo_long.csv", parse_dates=["data"])
    ibov = pd.read_csv(pasta / "ibov.csv", parse_dates=["Date"], index_col="Date")["ibov"]
    cal = ibov.index

    def matriz(col):
        return longo.pivot(index="data", columns="ticker", values=col).reindex(cal).sort_index(axis=1)

    precos, fechamento, volume = matriz("adj_close"), matriz("close"), matriz("volume")
    origem = dict.fromkeys(precos.columns, "yahoo")
    negocios = pd.DataFrame(index=cal)
    if usar_b3:
        b3 = pd.read_csv(pasta_b3 / "precos_b3_ajustados_long.csv", parse_dates=["data"])
        b3 = b3[~b3.ticker.isin(precos.columns)]

        def matriz_b3(col):
            return b3.pivot(index="data", columns="ticker", values=col).reindex(cal)

        precos = pd.concat([precos, matriz_b3("adj_close")], axis=1).sort_index(axis=1)
        fechamento = pd.concat([fechamento, matriz_b3("close")], axis=1).sort_index(axis=1)
        volume = pd.concat([volume, matriz_b3("volume")], axis=1).sort_index(axis=1)
        negocios = matriz_b3("negocios").sort_index(axis=1)
        origem |= dict.fromkeys(negocios.columns, "b3")
    for tk, inicio_valido in TRECHOS_INVALIDOS.items():
        precos.loc[precos.index < inicio_valido, tk] = np.nan
    return Brutos(precos, fechamento, volume, ibov, negocios, origem)


def carregar_cdi(caminho=RAIZ / "dados_brutos" / "bcb" / "cdi_sgs12.csv"):
    """cdi diario (sgs 12) em taxa decimal por dia util"""
    cdi = pd.read_csv(caminho, parse_dates=["data"], index_col="data")["cdi_pct_dia"]
    return (cdi / 100).rename("cdi")


def preencher(precos, limite=LIMITE_FFILL):
    """dia sem negociacao = preco do ultimo pregao, no max `limite` dias seguidos"""
    return precos.ffill(limit=limite)


def retornos_simples(precos):
    return precos.pct_change(fill_method=None).iloc[1:]


# limpeza
def limpar_extremos(ret, ret_ibov, negocios=None):
    """classifica cada |r| > LIMIAR_EXTREMO e trata os artefatos

    - degrau: |r| > LIMIAR_DEGRAU sem reversao e sem o mercado mexer, provavel evento
      n ajustado pelo yahoo, retorno vira 0. se tiver negocios da b3 e o dia teve
      >= LIMIAR_NEGOCIOS x mediana dos 60 pregoes antes, eh negociacao de verdade -> real
    - reversao: dia seguinte com sinal oposto e |r| > LIMIAR_REVERSAO. eh mantido, nos
      casos de 2011-2013 o volume foi de 2x a 8x a mediana, entao era negociacao real
    - real: o resto, mantido

    so usa o q ta em ret e negocios (ja cortados em t)
    retorna (ret_limpo, tabela_de_eventos)
    """
    ret = ret.copy()
    eventos = []
    datas = ret.index
    neg_rel = pd.DataFrame(index=datas)
    if negocios is not None and len(negocios.columns):
        neg = negocios.reindex(index=datas, columns=[c for c in negocios.columns if c in ret.columns])
        neg_rel = neg / neg.shift().rolling(60, min_periods=10).median()
    for tk in ret.columns:
        s = ret[tk]
        for d in s.index[s.abs() > LIMIAR_EXTREMO]:
            i = datas.get_loc(d)
            r = ret.at[d, tk]
            r_mkt = ret_ibov.get(d, np.nan)
            r_seg = ret[tk].iloc[i + 1] if i + 1 < len(datas) else np.nan
            sem_mercado = abs(r_mkt) < LIMIAR_MERCADO
            if sem_mercado and np.sign(r_seg) == -np.sign(r) and abs(r_seg) > LIMIAR_REVERSAO:
                tipo = "reversao"
            elif sem_mercado and abs(r) > LIMIAR_DEGRAU and not (
                    tk in neg_rel.columns and neg_rel.at[d, tk] >= LIMIAR_NEGOCIOS):
                tipo = "degrau"
                ret.at[d, tk] = 0.0
            else:
                tipo = "real"
            eventos.append({"data": d.date(), "ticker": tk, "retorno": r, "retorno_seguinte": r_seg,
                            "retorno_ibov": r_mkt, "classificacao": tipo})
    cols = ["data", "ticker", "retorno", "retorno_seguinte", "retorno_ibov", "classificacao"]
    return ret, pd.DataFrame(eventos, columns=cols)


# estimacao
@dataclass
class Estimativa:
    t: pd.Timestamp
    inicio: pd.Timestamp
    fim: pd.Timestamp                 # ultimo pregao usado (< t)
    tickers: list
    mu: pd.Series                     # anualizado
    sigma: pd.DataFrame               # anualizada, amostral
    sigma_lw: pd.DataFrame            # anualizada, Ledoit-Wolf
    shrinkage_lw: float
    retornos: pd.DataFrame            # retornos diários limpos usados
    exclusoes: pd.DataFrame           # ticker, motivo
    eventos: pd.DataFrame
    pares_correlacionados: pd.DataFrame = field(default_factory=pd.DataFrame)


def janela(df, t, anos=JANELA_ANOS):
    """linhas com data em [t - anos, t), nunca inclui t"""
    t = pd.Timestamp(t)
    return df.loc[(df.index >= t - pd.DateOffset(years=anos)) & (df.index < t)]


def filtro_correlacao(ret, liquidez, limiar=LIMIAR_CORRELACAO):
    """pra cada par com correlacao > limiar (do maior pro menor) tira o menos liquido"""
    corr = ret.corr()
    sup = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    sup.index.names = ["ativo_1", "ativo_2"]
    pares = sup.rename("correlacao").reset_index()
    pares = pares[pares.correlacao > limiar].sort_values("correlacao", ascending=False)
    removidos, registro = set(), []
    for _, p in pares.iterrows():
        a, b = p.ativo_1, p.ativo_2
        if a in removidos or b in removidos:
            continue
        sai = a if liquidez[a] < liquidez[b] else b
        removidos.add(sai)
        registro.append({**p.to_dict(), "liq_1": liquidez[a], "liq_2": liquidez[b], "removido": sai})
    return sorted(removidos), pd.DataFrame(registro)


def eh_psd(m, tol=1e-10):
    v = np.linalg.eigvalsh(np.asarray(m))
    return bool(v.min() >= -tol * max(1.0, v.max())), float(v.min()), float(v.max())


def estimar(brutos, t, anos=JANELA_ANOS):
    """mu e sigma anualizados com dados de [t - anos, t)"""
    t = pd.Timestamp(t)
    # corta antes de qualquer conta, nada de t pra frente entra
    precos = janela(brutos.precos, t, anos)
    fech = janela(brutos.fechamento, t, anos)
    vol = janela(brutos.volume, t, anos)
    ibov = janela(brutos.ibov, t, anos)
    neg = janela(brutos.negocios, t, anos)
    exclusoes = []

    # 1. historico completo na janela (depois de preencher ate LIMITE_FFILL dias)
    p = preencher(precos)
    incompletos = p.columns[p.isna().any()]
    exclusoes += [{"ticker": tk, "motivo": f"histórico incompleto ({int(p[tk].isna().sum())} pregões sem preço)"}
                  for tk in incompletos]
    p = p.drop(columns=incompletos)

    ret = retornos_simples(p)
    ret_ibov = retornos_simples(ibov)

    # 2. preco parado (sem negociacao de verdade)
    frac_zero = (ret == 0).mean()
    parados = frac_zero.index[frac_zero > LIMITE_RET_ZERO]
    exclusoes += [{"ticker": tk, "motivo": f"preço parado ({frac_zero[tk]:.0%} dos retornos nulos)"}
                  for tk in parados]
    ret = ret.drop(columns=parados)

    # 3. retornos extremos
    ret, eventos = limpar_extremos(ret, ret_ibov, neg)

    # 4. filtro de correlacao, desempata pela liquidez (volume financeiro mediano,
    #    ignorando volume zero q no yahoo antigo eh dado faltando)
    fin = (fech[ret.columns] * vol[ret.columns]).where(vol[ret.columns] > 0)
    liquidez = fin.median()
    removidos, pares = filtro_correlacao(ret, liquidez)
    exclusoes += [{"ticker": tk, "motivo": "correlação > 0,90 com ativo mais líquido"} for tk in removidos]
    ret = ret.drop(columns=removidos)

    mu = ret.mean() * PREGOES_ANO
    sigma = ret.cov() * PREGOES_ANO
    lw = LedoitWolf().fit(ret.values)
    sigma_lw = pd.DataFrame(lw.covariance_ * PREGOES_ANO, index=ret.columns, columns=ret.columns)

    return Estimativa(t=t, inicio=ret.index.min(), fim=ret.index.max(), tickers=list(ret.columns),
                      mu=mu, sigma=sigma, sigma_lw=sigma_lw, shrinkage_lw=float(lw.shrinkage_),
                      retornos=ret, exclusoes=pd.DataFrame(exclusoes, columns=["ticker", "motivo"]),
                      eventos=eventos, pares_correlacionados=pares)


def exportar_instancia(est, pasta, usar_lw=False, cenarios=True):
    """grava no formato do PortfolioData (c++): assets.csv (nome,retorno) e
    covariance.csv (N x N), sem cabecalho e na mesma ordem. com `cenarios` grava
    tambem scenarios.csv (S x N) com os retornos diarios da janela, q sao os cenarios do cvar"""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    sig = est.sigma_lw if usar_lw else est.sigma
    with open(pasta / "assets.csv", "w") as f:
        for tk in est.tickers:
            f.write(f"{tk},{est.mu[tk]:.10f}\n")
    np.savetxt(pasta / "covariance.csv", sig.loc[est.tickers, est.tickers].values, delimiter=",", fmt="%.12e")
    if cenarios:
        np.savetxt(pasta / "scenarios.csv", est.retornos[est.tickers].values, delimiter=",", fmt="%.10e")
