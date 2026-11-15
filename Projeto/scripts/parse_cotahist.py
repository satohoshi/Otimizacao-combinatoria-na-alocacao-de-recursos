"""leitura dos arquivos COTAHIST da b3, pras acoes q n tem no yahoo

layout conferido com o "Layout do arquivo - Cotacoes historicas" da b3 (rev 02,
05/10/2020): registros de 245 bytes, posicoes 1-based e precos (11)V99 (divide por 100)

o ativo eh seguido pelo isin (CODISI) e n so pelo ticker: pra cada ticker da lista
pega todos os registros dos isins em q ele aparece, e grava a linha do tempo
(primeiro/ultimo pregao por ticker x isin x nome) pra conferir troca de ticker
os renomeados sem mudanca societaria (EMENDAS) entram com o proprio ticker, a
emenda da serie eh feita no ajustar_b3.py

uso:
    python3 scripts/parse_cotahist.py
saidas em dados_brutos/b3/:
    precos_b3_long.csv     data, ticker, isin, nome, fechamento, volume_financeiro, qtd, negocios
                           (+ abertura, maxima, minima, especi, codbdi, fatcot, ativo)
    linha_do_tempo_b3.csv  primeiro/ultimo pregao por (ativo, ticker, isin, nome)
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / "dados_brutos" / "b3"

# as 17 acoes recuperadas e as 9 da segunda leva, por volume financeiro em 2013
# AMBV3 fica de fora pq a serie acaba em 08/11/2013, antes da primeira decisao
ALVOS = ["OGXP3", "CIEL3", "BRFS3", "BRML3", "HGTX3", "AEDU3", "BRPR3", "CTIP3", "JBSS3",
         "FIBR3", "LAME4", "ALLL3", "CRUZ3", "ENBR3", "GOLL4", "MMXM3", "ARTR3",
         "MPLU3", "ELPL4", "ODPV3", "GETI4", "BISA3", "LLXL3", "IMCH3", "BPHA3", "BBRK3"]
# antecessores/sucessores com outro isin mas mesma empresa, sem mudanca societaria
# OHLB3 -> ARTR3 (dez/2012), IMCH3 -> MEAL3 (31/03/2015), LLXL3 -> PRML3 (04/04/2014)
# pregoes seguidos, sem sobrepor
EMENDAS = {"ARTR3": ["OHLB3"], "IMCH3": ["MEAL3"], "LLXL3": ["PRML3"]}

# CODBDI aceitos no mercado a vista (TPMERC 010). alem do lote padrao (02) a b3 muda
# o papel de bdi em situacao especial, sem isso a serie de empresa em recuperacao judicial
# seria cortada
#   05 sancionadas, 06 concordatarias, 07 recuperacao extrajudicial,
#   08 recuperacao judicial, 09 RAET, 11 intervencao, 58 outros
BDI_ACEITOS = {"02", "05", "06", "07", "08", "09", "11", "58"}

# (nome, inicio, fim) em posicoes 1-based do layout
CAMPOS = [("tipreg", 1, 2), ("data", 3, 10), ("codbdi", 11, 12), ("codneg", 13, 24),
          ("tpmerc", 25, 27), ("nomres", 28, 39), ("especi", 40, 49),
          ("preabe", 57, 69), ("premax", 70, 82), ("premin", 83, 95), ("premed", 96, 108),
          ("preult", 109, 121), ("totneg", 148, 152), ("quatot", 153, 170),
          ("voltot", 171, 188), ("fatcot", 211, 217), ("codisi", 231, 242), ("dismes", 243, 245)]
PRECOS = ("preabe", "premax", "premin", "premed", "preult")
INTEIROS = ("totneg", "quatot", "fatcot", "dismes")


def ler_registro(linha):
    """registro tipo 01 (str de 245 colunas) -> dict ja convertido

    precos (11)V99 e VOLTOT (16)V99 dividem por 100, e os precos ainda dividem por
    FATCOT (cotacao por lote de FATCOT acoes)"""
    r = {nome: linha[i - 1:f].strip() for nome, i, f in CAMPOS}
    for c in INTEIROS:
        r[c] = int(r[c])
    fat = r["fatcot"] or 1
    for c in PRECOS:
        r[c] = int(r[c]) / 100 / fat
    r["voltot"] = int(r["voltot"]) / 100
    r["data"] = pd.Timestamp(r["data"])
    return r


def linhas_vista(caminho_zip):
    """linhas brutas (bytes) dos registros 01 do mercado a vista nos bdis aceitos"""
    with zipfile.ZipFile(caminho_zip) as z:
        for nome in z.namelist():
            for bruto in z.open(nome):
                if bruto[:2] == b"01" and bruto[24:27] == b"010" and bruto[10:12].decode() in BDI_ACEITOS:
                    yield bruto


def main():
    zips = sorted((PASTA / "zip").glob("COTAHIST_A*.ZIP"))
    if not zips:
        sys.exit(f"nenhum COTAHIST em {PASTA / 'zip'}; rode scripts/coletar_cotahist.py")
    procurados = set(ALVOS) | {a for v in EMENDAS.values() for a in v}

    # passada 1: (ticker, isin, nome) de todos os papeis a vista, so fatiando bytes
    chaves = set()
    for z in zips:
        for b in linhas_vista(z):
            chaves.add((b[12:24].decode("latin-1").strip(), b[230:242].decode("latin-1"),
                        b[27:39].decode("latin-1").strip()))
    chaves = pd.DataFrame(sorted(chaves), columns=["codneg", "codisi", "nomres"])
    faltando = sorted(procurados - set(chaves.codneg))
    if faltando:
        print(f"NÃO ENCONTRADOS no COTAHIST: {faltando}", file=sys.stderr)
    dono = {}
    for alvo in ALVOS:
        for tk in [alvo] + EMENDAS.get(alvo, []):
            for isin in chaves.codisi[chaves.codneg == tk]:
                dono[isin] = alvo

    # passada 2: todos os registros dos isins seguidos (inclui outros tickers do mesmo isin)
    alvo_isin = {i.encode() for i in dono}
    regs = []
    for z in zips:
        n = len(regs)
        regs += [ler_registro(b.decode("latin-1").rstrip("\r\n")) for b in linhas_vista(z)
                 if b[230:242] in alvo_isin]
        print(f"{z.name}: {len(regs) - n} registros", flush=True)
    sel = pd.DataFrame(regs)
    sel["ativo"] = sel.codisi.map(dono)

    dup = sel.duplicated(["ativo", "codneg", "data"], keep=False)
    if dup.any():   # n acontece em 2008-2026, so por garantia
        sys.exit(f"pregões duplicados por ticker:\n{sel[dup].head(20)}")

    longo = pd.DataFrame({
        "data": sel.data.dt.date, "ticker": sel.codneg, "isin": sel.codisi, "nome": sel.nomres,
        "fechamento": sel.preult, "volume_financeiro": sel.voltot, "qtd": sel.quatot,
        "negocios": sel.totneg, "abertura": sel.preabe, "maxima": sel.premax, "minima": sel.premin,
        "especi": sel.especi, "codbdi": sel.codbdi, "fatcot": sel.fatcot, "ativo": sel.ativo,
    }).sort_values(["ativo", "data", "ticker"])
    longo.to_csv(PASTA / "precos_b3_long.csv", index=False)

    tl = (longo.groupby(["ativo", "ticker", "isin", "nome"])
          .agg(primeiro=("data", "min"), ultimo=("data", "max"), pregoes=("data", "size"),
               bdis=("codbdi", lambda s: " ".join(sorted(set(s)))))
          .reset_index().sort_values(["ativo", "primeiro"]))
    tl.to_csv(PASTA / "linha_do_tempo_b3.csv", index=False)

    # mesmo ticker com mais de um isin ou nome no periodo
    reuso = chaves[chaves.codneg.isin(procurados)].groupby("codneg").agg(
        n_isin=("codisi", "nunique"), n_nome=("nomres", "nunique"),
        nomes=("nomres", lambda s: " | ".join(sorted(set(s)))))
    pd.set_option("display.width", 200)
    print("\nLinha do tempo (ticker x ISIN):\n" + tl.to_string(index=False))
    print("\nConferência de reuso de código:\n" + reuso.to_string())


if __name__ == "__main__":
    main()
