"""base consolidada: yahoo + cotahist da b3 nos mesmos arquivos

junta as 73 series do yahoo e as 26 do cotahist com a mesma regra do dados.carregar
(yahoo tem prioridade, tudo no calendario do ibov), marca a origem de cada serie e
cria um inventario dos 100 tickers do sua_lista.txt. n mexe em dados_brutos/yahoo/
nem dados_brutos/b3/, eh so uma copia pra consulta

os TRECHOS_INVALIDOS do dados.py (TIMP3 antes de 01/08/2011) n sao aplicados aqui,
a base guarda o dado original e o trecho eh tirado no dados.carregar

uso:
    python3 scripts/consolidar_bases.py
saidas em dados_brutos/consolidado/:
    precos_long.csv        data, ticker, origem, ticker_fonte, isin, close, adj_close,
                           volume, negocios, ajuste_proventos
    matriz_adj_close.csv   preco ajustado (data x ticker), calendario do ibov
    inventario_ativos.csv  uma linha por ticker (origem, periodo, ajustes, motivo do
                           fim da serie, se entra em 2014-01-02)
    metadados.json
"""
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dados  # noqa: E402

RAIZ = dados.RAIZ
SAIDA = RAIZ / "dados_brutos" / "consolidado"
T_REFERENCIA = "2014-01-02_b3"      # data usada pra coluna de elegibilidade
COLUNAS = ["data", "ticker", "origem", "ticker_fonte", "isin", "close", "adj_close", "volume",
           "negocios", "ajuste_proventos"]


def base_longa():
    """formato longo das duas fontes, so com pregoes do ibov"""
    cal = pd.read_csv(dados.BRUTOS / "ibov.csv", parse_dates=["Date"]).Date
    y = pd.read_csv(dados.BRUTOS / "precos_yahoo_long.csv", parse_dates=["data"])
    y = y.rename(columns={"ticker_yahoo": "ticker_fonte"}).assign(
        origem="yahoo", isin=pd.NA, negocios=pd.NA, ajuste_proventos=True)
    b = pd.read_csv(dados.BRUTOS_B3 / "precos_b3_ajustados_long.csv", parse_dates=["data"])
    b = b[~b.ticker.isin(y.ticker.unique())]              # yahoo tem prioridade
    b = b.rename(columns={"ticker_negociado": "ticker_fonte"}).assign(origem="b3", ajuste_proventos=False)
    longo = pd.concat([y[COLUNAS], b[COLUNAS]], ignore_index=True)
    return longo[longo.data.isin(cal)].sort_values(["ticker", "data"]).reset_index(drop=True)


def inventario(longo):
    lista = (RAIZ / "sua_lista.txt").read_text().split()
    falhas = pd.read_csv(dados.BRUTOS / "falhas.csv").set_index("ticker")
    qual = pd.read_csv(dados.BRUTOS_B3 / "qualidade_b3.csv").set_index("ticker")
    ref = RAIZ / "data" / "processado" / T_REFERENCIA
    elegiveis = set(pd.read_csv(ref / "mu.csv").ticker)
    exclusoes = pd.read_csv(ref / "exclusoes.csv").set_index("ticker").motivo
    resumo = longo.dropna(subset=["adj_close"]).groupby("ticker").agg(
        origem=("origem", "first"), ticker_fonte=("ticker_fonte", lambda s: " -> ".join(dict.fromkeys(s))),
        isin=("isin", lambda s: " -> ".join(dict.fromkeys(s.dropna())) or pd.NA),
        primeiro_pregao=("data", "min"), ultimo_pregao=("data", "max"), n_pregoes=("data", "size"),
        ajuste_proventos=("ajuste_proventos", "first"))
    linhas = []
    for tk in lista:
        if tk in resumo.index:
            r = resumo.loc[tk]
            b3 = qual.loc[tk] if r.origem == "b3" else None
            linhas.append({
                "ticker": tk, **r.to_dict(),
                "primeiro_pregao": r.primeiro_pregao.date(), "ultimo_pregao": r.ultimo_pregao.date(),
                "eventos_aplicados": b3.eventos_aplicados if b3 is not None else "ajuste do Yahoo",
                "causa_fim": f"{b3.causa_fim}: {b3.detalhe_fim}" if b3 is not None else pd.NA,
                "situacao_2014": "elegível" if tk in elegiveis else exclusoes.get(tk, "fora da janela"),
            })
        else:
            linhas.append({"ticker": tk, "origem": "sem dados", "ticker_fonte": falhas.ticker_yahoo.get(tk, pd.NA),
                           "situacao_2014": "sem dados (Yahoo e COTAHIST não coletados)"})
    return pd.DataFrame(linhas)


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    longo = base_longa()
    longo.to_csv(SAIDA / "precos_long.csv", index=False)
    cal = pd.read_csv(dados.BRUTOS / "ibov.csv", parse_dates=["Date"], index_col="Date").index
    matriz = longo.pivot(index="data", columns="ticker", values="adj_close").reindex(cal).sort_index(axis=1)
    matriz.to_csv(SAIDA / "matriz_adj_close.csv", index_label="Date")
    inv = inventario(longo)
    inv.to_csv(SAIDA / "inventario_ativos.csv", index=False)

    contagem = inv.origem.value_counts().to_dict()
    meta = {
        "gerado_em": date.today().isoformat(),
        "fontes": {"yahoo": "Yahoo Finance via yfinance (dados_brutos/yahoo/), preço ajustado por proventos e eventos",
                   "b3": "COTAHIST da B3 (dados_brutos/b3/), ajustado só por eventos em ações (D23, D24)"},
        "prioridade": "Yahoo; B3 só para tickers ausentes no Yahoo",
        "calendario": "pregões do Ibovespa (^BVSP)",
        "periodo": [str(longo.data.min().date()), str(longo.data.max().date())],
        "n_tickers_lista": len(inv), "n_por_origem": contagem,
        "n_linhas": len(longo), "trechos_invalidos_nao_aplicados": dados.TRECHOS_INVALIDOS,
        "elegiveis_" + T_REFERENCIA: int((inv.situacao_2014 == "elegível").sum()),
    }
    (SAIDA / "metadados.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
