"""baixa o cdi diario (serie 12 do sgs do bcb, % ao dia)

a api do sgs so deixa pegar 10 anos de serie diaria por vez, entao baixa por ano
so baixa se o arquivo ainda n existir

uso:
    python3 scripts/coletar_cdi.py [--forcar]
"""
import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "dados_brutos" / "bcb" / "cdi_sgs12.csv"
URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
INICIO, FIM = 2008, 2026


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcar", action="store_true")
    if SAIDA.exists() and not ap.parse_args().forcar:
        print(f"{SAIDA} já existe; use --forcar para baixar de novo")
        return
    partes = []
    for ano in range(INICIO, FIM + 1):
        r = requests.get(URL, params={"formato": "json", "dataInicial": f"01/01/{ano}",
                                      "dataFinal": f"31/12/{ano}"}, timeout=60)
        r.raise_for_status()
        partes.append(pd.DataFrame(r.json()))
        print(ano, len(partes[-1]))
    df = pd.concat(partes, ignore_index=True)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["cdi_pct_dia"] = df["valor"].astype(float)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    df[["data", "cdi_pct_dia"]].drop_duplicates("data").sort_values("data").to_csv(SAIDA, index=False)
    (SAIDA.parent / "metadados.json").write_text(json.dumps(
        {"fonte": "Banco Central do Brasil, SGS série 12 (CDI, % a.d.)", "url": URL,
         "data_download": date.today().isoformat(), "n_obs": len(df)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
