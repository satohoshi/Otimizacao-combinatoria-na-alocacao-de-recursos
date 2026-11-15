"""coleta os precos diarios no yahoo finance (yfinance)

baixa o periodo todo [INICIO, FIM) numa chamada so por ticker, pra o fator de
ajuste (dividendo, jcp, desdobramento, grupamento) ser o mesmo na serie toda
juntar series baixadas em dias diferentes mistura fatores de ajuste

universo: sua_lista.txt (acoes mais liquidas de 2013). o ticker de 2013 -> ticker
atual no yahoo (ex: CCRO3 -> MOTV3) vem do saida_yahoo/qualidade_por_ticker.csv

uso:
    python3 scripts/coletar_yahoo.py            # salva em dados_brutos/yahoo/
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

RAIZ = Path(__file__).resolve().parents[1]
INICIO = "2008-12-01"          # janela antes de 2014 + folga
FIM = "2026-10-08"             # exclusivo
SAIDA = RAIZ / "dados_brutos" / "yahoo"


def mapa_tickers():
    """ticker de 2013 -> ticker no yahoo (sem o .SA)"""
    lista = [t.strip() for t in (RAIZ / "sua_lista.txt").read_text().split() if t.strip()]
    q = pd.read_csv(RAIZ / "saida_yahoo" / "qualidade_por_ticker.csv")
    mapa = dict(zip(q.ticker, q.ticker_yahoo))
    # os q falharam na coleta original tenta com o proprio nome
    return {t: mapa.get(t, t) for t in lista}


def baixar(ticker_yahoo):
    h = yf.Ticker(ticker_yahoo).history(start=INICIO, end=FIM, auto_adjust=False,
                                        actions=True)
    if h.empty:
        return h
    h.index = h.index.tz_localize(None).normalize()
    return h


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    mapa = mapa_tickers()
    linhas, falhas = [], []
    for i, (tk, tky) in enumerate(mapa.items(), 1):
        h = pd.DataFrame()
        for tentativa in range(3):
            try:
                h = baixar(tky + ".SA")
                break
            except Exception as e:  # erro de rede, tenta de novo
                print(f"  {tk}: erro {e!r}, tentativa {tentativa + 1}", file=sys.stderr)
                time.sleep(2)
        if h.empty:
            falhas.append({"ticker": tk, "ticker_yahoo": tky, "motivo": "sem dados no Yahoo"})
            print(f"[{i}/{len(mapa)}] {tk}: sem dados")
            continue
        df = pd.DataFrame({
            "data": h.index, "ticker": tk, "ticker_yahoo": tky,
            "close": h["Close"].values, "adj_close": h["Adj Close"].values,
            "dividendos": h["Dividends"].values, "splits": h["Stock Splits"].values,
            "volume": h["Volume"].values,
        })
        linhas.append(df)
        print(f"[{i}/{len(mapa)}] {tk} ({tky}): {len(df)} obs, {df.data.min().date()}")

    longo = pd.concat(linhas, ignore_index=True)
    longo.to_csv(SAIDA / "precos_yahoo_long.csv", index=False)
    longo.pivot(index="data", columns="ticker", values="adj_close").sort_index() \
        .to_csv(SAIDA / "matriz_adj_close.csv", index_label="Date")
    pd.DataFrame(falhas, columns=["ticker", "ticker_yahoo", "motivo"]) \
        .to_csv(SAIDA / "falhas.csv", index=False)

    ib = baixar("^BVSP")
    ib[["Close"]].rename(columns={"Close": "ibov"}).to_csv(SAIDA / "ibov.csv", index_label="Date")

    meta = {"fonte": "Yahoo Finance via yfinance", "yfinance": yf.__version__,
            "data_download": date.today().isoformat(), "inicio": INICIO, "fim_exclusivo": FIM,
            "auto_adjust": False, "coluna_ajustada": "Adj Close",
            "n_tickers_lista": len(mapa), "n_baixados": longo.ticker.nunique(),
            "n_falhas": len(falhas)}
    (SAIDA / "metadados.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
