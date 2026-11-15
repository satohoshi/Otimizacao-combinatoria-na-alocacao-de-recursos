"""baixa os arquivos anuais COTAHIST da b3

baixa COTAHIST_A<ANO>.ZIP de ANO_INICIO a ANO_FIM pra dados_brutos/b3/zip/
n baixa de novo se ja existe e abre como zip (a menos q passe --forcar). se a b3
bloquear ou mudar o link ele para e lista o q tem q baixar na mao

tambem salva em dados_brutos/b3/eventos_api/ a resposta da api publica de eventos
corporativos da b3 (GetListedSupplementCompany) de cada empresa. pras deslistadas
vem incompleto, entao so serve pra confirmar eventos

uso:
    python3 scripts/coletar_cotahist.py              # 2008 a 2026
    python3 scripts/coletar_cotahist.py --forcar     # baixa de novo
"""
import argparse
import base64
import hashlib
import json
import sys
import zipfile
from datetime import date
from pathlib import Path

import requests

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "dados_brutos" / "b3" / "zip"
URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{ano}.ZIP"
ANO_INICIO, ANO_FIM = 2008, 2026     # mesmo periodo do yahoo (desde 2008-12)
URL_EVENTOS = ("https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/"
               "GetListedSupplementCompany/{consulta}")
# codigo do emissor (4 letras) das 17 acoes recuperadas e da OHLB3
EMISSORES = ["OGXP", "CIEL", "BRFS", "BRML", "HGTX", "AEDU", "BRPR", "CTIP", "JBSS", "FIBR",
             "LAME", "ALLL", "CRUZ", "ENBR", "GOLL", "MMXM", "ARTR", "OHLB"]


def zip_valido(caminho):
    try:
        with zipfile.ZipFile(caminho) as z:
            return z.testzip() is None and len(z.namelist()) > 0
    except (zipfile.BadZipFile, OSError):
        return False


def baixar(ano, destino):
    r = requests.get(URL.format(ano=ano), timeout=120, stream=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    tmp = destino.with_suffix(".part")
    with open(tmp, "wb") as f:
        for bloco in r.iter_content(1 << 20):
            f.write(bloco)
    if not zip_valido(tmp):
        tmp.unlink()
        raise RuntimeError("resposta não é um ZIP válido (URL mudou ou bloqueio)")
    tmp.rename(destino)


def baixar_eventos(forcar=False):
    """eventos corporativos (em acoes e em dinheiro) por emissor, do jeito q a b3 publica"""
    pasta = SAIDA.parent / "eventos_api"
    pasta.mkdir(parents=True, exist_ok=True)
    for emp in EMISSORES:
        destino = pasta / f"{emp}.json"
        if destino.exists() and not forcar:
            continue
        consulta = base64.b64encode(json.dumps({"issuingCompany": emp, "language": "pt-br"}).encode()).decode()
        r = requests.get(URL_EVENTOS.format(consulta=consulta), timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        destino.write_bytes(r.content)
        print(f"eventos {emp}: {len(r.content)} bytes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcar", action="store_true", help="baixa de novo mesmo se já existir")
    args = ap.parse_args()
    SAIDA.mkdir(parents=True, exist_ok=True)

    falhas, arquivos = [], {}
    for ano in range(ANO_INICIO, ANO_FIM + 1):
        destino = SAIDA / f"COTAHIST_A{ano}.ZIP"
        if destino.exists() and zip_valido(destino) and not args.forcar:
            print(f"{ano}: já existe")
        else:
            try:
                baixar(ano, destino)
                print(f"{ano}: baixado ({destino.stat().st_size / 1e6:.1f} MB)")
            except Exception as e:
                falhas.append(ano)
                print(f"{ano}: FALHOU ({e})", file=sys.stderr)
                continue
        arquivos[destino.name] = hashlib.sha256(destino.read_bytes()).hexdigest()

    if falhas:
        print("\nBaixe manualmente em https://www.b3.com.br (Market Data > Séries históricas)\n"
              f"e coloque em {SAIDA}:\n" + "\n".join(f"  COTAHIST_A{a}.ZIP" for a in falhas),
              file=sys.stderr)
        sys.exit(1)

    baixar_eventos(args.forcar)
    meta = {"fonte": "B3, séries históricas COTAHIST", "url": URL, "url_eventos": URL_EVENTOS,
            "data_download": date.today().isoformat(),
            "anos": [ANO_INICIO, ANO_FIM], "sha256": arquivos}
    (SAIDA.parent / "metadados_zip.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
