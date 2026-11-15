#!/usr/bin/env python3
"""
baixa do yahoo os precos das acoes da b3 ja ajustados (dividendo, jcp,
desdobramento e grupamento, coluna adj_close) a partir de uma lista de tickers

entrada (escolher uma):
  --tickers lista.txt                       um ticker por linha (ex: PETR4)
  --cobertura saida_cotahist/cobertura_por_ticker.csv --top 120
        usa o arquivo do baixar_cotahist_ipca.py e pega as N com maior volume mediano

saidas (pasta saida_yahoo/):
  precos_yahoo_long.csv        data, ticker, close, adj_close, dividendos, splits, volume
  matriz_adj_close.csv         data x ticker (preco ajustado)
  ibov.csv                     ^BVSP (calendario de referencia)
  qualidade_por_ticker.csv     primeira data, cobertura, saltos, flag historico_ok
  falhas.csv                   o q n achou no yahoo

obs:
  - o close do yahoo ja vem ajustado por desdobramento/grupamento mas n por dividendo,
    o adj_close inclui dividendo (retorno total). salva os dois pra conferir
  - acao q saiu da bolsa normalmente n tem no yahoo (vira falha)
  - ticker q mudou de codigo eh trocado pelo atual segundo o MAPA
"""

import argparse
import os
import sys
import time
from datetime import date

import pandas as pd
import yfinance as yf

# codigo antigo (cotahist) -> codigo atual no yahoo
MAPA = {
    "BVMF3": "B3SA3",   # BM&FBovespa -> B3
    "KROT3": "COGN3",   # Kroton -> Cogna
    "TBLE3": "EGIE3",   # Tractebel -> Engie Brasil
    "ESTC3": "YDUQ3",   # Estacio -> YDUQS
    "PCAR4": "PCAR3",   # Pao de Acucar
    "AMBV4": "ABEV3",   # Ambev
    "VIVT4": "VIVT3",   # Telefonica Brasil
    "TIMP3": "TIMS3",   # TIM
    "ELET3": "AXIA3",   # Eletrobras -> Axia (verificar)
    "CCRO3": "MOTV3",   # CCR -> Motiva (verificar)
"EMBR3": "EMBJ3", "DTEX3": "DXCO3", "HRTP3": "PRIO3", "MPXE3": "ENEV3",
"BTOW3": "AMER3", "ALSC3": "ALOS3", "IGTA3": "IGTI11", "MRFG3": "MBRF3",
}


def ler_tickers(args) -> list:
    if args.tickers:
        with open(args.tickers, encoding="utf-8") as f:
            return [l.strip().upper() for l in f if l.strip() and not l.startswith("#")]
    if args.cobertura:
        c = pd.read_csv(args.cobertura)
        c = c.sort_values("volume_mediano", ascending=False).head(args.top)
        return c["ticker"].str.upper().tolist()
    sys.exit("Informe --tickers lista.txt ou --cobertura arquivo.csv")


def baixar(ticker_yahoo: str, inicio: str, fim: str, tentativas: int = 3):
    """devolve o dataframe diario ou None, tenta de novo se der erro de rede/limite"""
    for i in range(tentativas):
        try:
            h = yf.Ticker(ticker_yahoo).history(
                start=inicio, end=fim, auto_adjust=False, actions=True,
                raise_errors=False,
            )
            if h is None or h.empty:
                return None  # nao encontrado
            h = h.copy()
            h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
            h = h[~h.index.duplicated(keep="last")]
            return h
        except Exception as e:  # noqa: BLE001
            espera = 5 * (3 ** i)
            print(f"    erro ({type(e).__name__}); tentando de novo em {espera}s")
            time.sleep(espera)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Precos ajustados do Yahoo Finance (B3)")
    ap.add_argument("--tickers")
    ap.add_argument("--cobertura")
    ap.add_argument("--top", type=int, default=120)
    ap.add_argument("--inicio", default="2013-12-01")
    ap.add_argument("--fim", default=date.today().isoformat())
    ap.add_argument("--saida-dir", default="saida_yahoo")
    ap.add_argument("--pausa", type=float, default=0.6,
                    help="segundos entre downloads (evita bloqueio do Yahoo)")
    args = ap.parse_args()

    os.makedirs(args.saida_dir, exist_ok=True)
    tickers = ler_tickers(args)
    print(f"{len(tickers)} tickers para baixar de {args.inicio} a {args.fim}")

    # calendario de referencia: ibov
    ibov = baixar("^BVSP", args.inicio, args.fim)
    if ibov is None:
        sys.exit("Nao consegui baixar o ^BVSP. Verifique a internet e tente de novo.")
    ibov[["Close"]].rename(columns={"Close": "ibov"}).to_csv(
        os.path.join(args.saida_dir, "ibov.csv"), date_format="%Y-%m-%d")
    calendario = ibov.index

    partes, falhas, qual = [], [], []
    for k, t in enumerate(tickers, 1):
        ty = MAPA.get(t, t)
        rotulo = f"{t} -> {ty}" if ty != t else t
        print(f"[{k}/{len(tickers)}] {rotulo}")
        h = baixar(ty + ".SA", args.inicio, args.fim)
        time.sleep(args.pausa)
        if h is None:
            falhas.append({"ticker": t, "ticker_yahoo": ty, "motivo": "sem dados no Yahoo"})
            continue

        d = pd.DataFrame({
            "close": h["Close"],
            "adj_close": h["Adj Close"] if "Adj Close" in h else h["Close"],
            "dividendos": h.get("Dividends", 0.0),
            "splits": h.get("Stock Splits", 0.0),
            "volume": h["Volume"],
        })
        d.index.name = "data"
        d = d.reset_index()
        d.insert(1, "ticker", t)
        d.insert(2, "ticker_yahoo", ty)
        partes.append(d)

        s = h["Adj Close"] if "Adj Close" in h else h["Close"]
        s = s.dropna()
        s = s[s > 0]
        primeira = s.index.min()
        cal = calendario[calendario >= primeira]
        cobertura = float(s.index.isin(cal).sum() / len(cal)) if len(cal) else 0.0
        ret = s.pct_change().dropna()
        qual.append({
            "ticker": t, "ticker_yahoo": ty,
            "primeira_data": primeira, "ultima_data": s.index.max(),
            "n_obs": len(s), "cobertura": round(cobertura, 4),
            "saltos_acima_20pct": int((ret.abs() > 0.20).sum()),
            "n_eventos_dividendo": int((d["dividendos"] > 0).sum()),
            "n_splits": int((d["splits"] > 0).sum()),
            "historico_ok": bool(primeira <= pd.Timestamp("2014-01-15") and cobertura >= 0.95),
        })

    if not partes:
        print("Nada foi baixado.")
        pd.DataFrame(falhas).to_csv(os.path.join(args.saida_dir, "falhas.csv"), index=False)
        return 1

    longo = pd.concat(partes, ignore_index=True).sort_values(["ticker", "data"])
    longo.to_csv(os.path.join(args.saida_dir, "precos_yahoo_long.csv"),
                 index=False, date_format="%Y-%m-%d", float_format="%.6f")

    matriz = longo.pivot_table(index="data", columns="ticker", values="adj_close", aggfunc="last")
    matriz = matriz.reindex(calendario).sort_index()  # alinha no calendario do ibov
    matriz.to_csv(os.path.join(args.saida_dir, "matriz_adj_close.csv"),
                  date_format="%Y-%m-%d", float_format="%.6f")

    q = pd.DataFrame(qual)
    q.to_csv(os.path.join(args.saida_dir, "qualidade_por_ticker.csv"),
             index=False, date_format="%Y-%m-%d")
    pd.DataFrame(falhas, columns=["ticker", "ticker_yahoo", "motivo"]).to_csv(
        os.path.join(args.saida_dir, "falhas.csv"), index=False)

    print("\n=== RESUMO ===")
    print(f"Baixados: {len(q)}   Falharam: {len(falhas)}")
    print(f"Com historico adequado desde 2014 e cobertura >= 95%: {int(q['historico_ok'].sum())}")
    print(f"Arquivos em '{args.saida_dir}/'. Veja falhas.csv e qualidade_por_ticker.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())