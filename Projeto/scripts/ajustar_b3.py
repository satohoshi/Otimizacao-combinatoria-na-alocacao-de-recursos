"""ajuste por eventos corporativos das series do cotahist

o cotahist vem com preco bruto, entao aqui:
1. junta os tickers renomeados (parse_cotahist.EMENDAS) numa serie so;
2. lista os candidatos a evento (eventos_candidatos.csv): pregao q comeca marcacao
   "ex" no ESPECI com salto de preco e todo |retorno| > 20% sem marcacao;
3. aplica pra tras so os eventos q mudam a qtd de acoes ou distribuem valor;
4. grava a matriz ajustada (mesmo formato do yahoo), o formato longo e a qualidade.

dividendo e jcp comum n sao ajustados, o cotahist n tem o valor e a api da b3
n publica pras deslistadas

uso:
    python3 scripts/ajustar_b3.py
saidas em dados_brutos/b3/:
    eventos_candidatos.csv, matriz_adj_close_b3.csv, precos_b3_ajustados_long.csv, qualidade_b3.csv
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_cotahist import ALVOS, PASTA, RAIZ  # noqa: E402

INICIO, FIM = "2008-12-01", "2026-10-08"     # mesmo periodo do yahoo (fim exclusivo)
LIMIAR_SALTO = 0.04          # |log(razao)| abaixo disso = ex sem salto
LIMIAR_EXTREMO = 0.20        # mesmo limiar do dados.py
TOLERANCIA_FATOR = 0.07      # |log| max entre a razao e o fator redondo
LIMIAR_DISTRIBUICAO = 0.10   # queda (descontando ibov) acima disso numa data ex = distribuicao grande
PRECO_CENTAVOS = 2.0         # abaixo disso o tick de 0.01 atrapalha, usa mediana de 5 pregoes

# fatores redondos de qtd de acoes (depois / antes):
# desdobramento, grupamento e bonificacao em multiplos de 5%
FATORES = sorted({2, 3, 4, 5, 10, 20, 100} | {1 / k for k in (2, 3, 4, 5, 10, 20, 25, 40, 50, 100)}
                 | {round(1 + b / 100, 2) for b in range(5, 55, 5)})

# eventos confirmados em fonte externa: (tipo, quantidade, fonte)
#   acoes: quantidade = acoes depois / antes (preco anterior divide por isso)
#   provento: quantidade = R$ por acao na data ex
CONFIRMADOS = {
    ("AEDU3", "2013-05-02"): ("acoes", 3.0, "API B3: desdobramento 200%"),
    ("BRFS3", "2010-04-01"): ("acoes", 2.0, "API B3: desdobramento 100%"),
    ("BRML3", "2010-09-24"): ("acoes", 2.0, "API B3: desdobramento 100%"),
    ("BRML3", "2017-05-02"): ("acoes", 1.15, "API B3: bonificação 15%"),
    ("CIEL3", "2011-05-02"): ("acoes", 1.2, "Investing.com: 1,2:1"),
    ("CIEL3", "2011-05-30"): ("acoes", 1 / 3, "API B3: grupamento 3:1"),
    ("CIEL3", "2012-04-23"): ("acoes", 1.2, "Investing.com: 1,2:1"),
    ("CIEL3", "2013-04-29"): ("acoes", 1.2, "Investing.com: 1,2:1"),
    ("CIEL3", "2014-04-01"): ("acoes", 2.0, "Investing.com: 2:1"),
    ("CIEL3", "2015-04-13"): ("acoes", 1.2, "Investing.com: 1,2:1"),
    ("CIEL3", "2016-04-11"): ("acoes", 1.2, "Investing.com: 1,2:1"),
    ("CIEL3", "2017-04-13"): ("acoes", 1.2, "API B3: bonificação 20%"),
    ("LAME4", "2016-03-31"): ("acoes", 1.2, "Suno: bonificação 20%"),
    ("MMXM3", "2016-05-02"): ("acoes", 1 / 25, "Status Invest/Suno: grupamento 25:1"),
    ("BRPR3", "2014-06-16"): ("provento", 5.50, "Release 2T14: dividendo extraordinário R$ 5,50/ação"),
    ("BRPR3", "2023-02-24"): ("acoes", 1 / 40, "AGE 24/01/2023: grupamento 40:1"),
}

# como cada serie termina (classe, descricao), fonte: api de eventos da b3
# (eventos_api/*.json) quando tem
FIM_SERIE = {
    "OGXP3": ("falência/recuperação judicial", "RJ em 2013; código substituído por DMMO3 (Dommo, novo ISIN)"),
    "MMXM3": ("falência/recuperação judicial", "RJ; falência mantida em 05/2021"),
    "AEDU3": ("incorporação", "incorporada pela Kroton (API B3: KROT3, 03/07/2014)"),
    "ALLL3": ("incorporação", "incorporada pela Rumo (RUMO3, 04/2015)"),
    "ARTR3": ("fechamento de capital", "saída do Novo Mercado e OPA (2016)"),
    "BRFS3": ("incorporação", "fusão com Marfrig (API B3: MRFG3 -> MBRF3, 22/09/2025)"),
    "BRML3": ("incorporação", "incorporada pela Aliansce Sonae (API B3, 06/01/2023; hoje ALOS3)"),
    "BRPR3": ("fechamento de capital", "OPA de cancelamento de registro (2023)"),
    "CIEL3": ("fechamento de capital", "OPA; resgate total (API B3, 23/09/2024)"),
    "CRUZ3": ("fechamento de capital", "OPA da BAT (2015)"),
    "CTIP3": ("incorporação", "incorporada pela BM&FBovespa (B3SA3, 2017)"),
    "ENBR3": ("fechamento de capital", "OPA da EDP (2023)"),
    "FIBR3": ("incorporação", "incorporada pela Suzano (SUZB3, 01/2019)"),
    "GOLL4": ("migração de ticker", "reestruturação de capital (Chapter 11); passou a GOLL54 com novo ISIN"),
    "HGTX3": ("incorporação", "incorporada pelo Grupo Soma (SOMA3, 2021)"),
    "JBSS3": ("migração de ticker", "listagem principal nos EUA; BDR JBSS32 (API B3, 06/06/2025)"),
    "LAME4": ("incorporação", "incorporada pela Americanas S.A. (AMER3, 21/01/2022)"),
    # segunda leva
    "MPLU3": ("fechamento de capital", "OPA da LATAM (2019)"),
    "ELPL4": ("migração de ticker", "migração ao Novo Mercado com conversão das PN em ON (ELPL3, 11/2017)"),
    "ODPV3": ("incorporação", "incorporada pela Bradesco Saúde (último pregão como BRADSAUDE, 04/05/2026)"),
    "GETI4": ("migração de ticker", "migração para units TIET11 (AES Tietê Energia, 01/2016)"),
    "BISA3": ("fechamento de capital", "OPA da Brookfield (01/2015)"),
    "LLXL3": ("fechamento de capital", "renomeada Prumo (PRML3, 04/2014; série emendada); OPA (04/2018)"),
    "IMCH3": ("em negociação", "renomeada IMC S/A (MEAL3, 31/03/2015; série emendada)"),
    "BPHA3": ("falência/recuperação judicial", "RJ em 2018 (BDI 08); cancelamento do registro (06/2019)"),
    "BBRK3": ("migração de ticker", "grupamento e troca para NEXP3 (Nexpe, novo ISIN, 06/2022); não emendada"),
}


def fator_redondo(q):
    """fator m (de FATORES) mais perto de 1/q e o erro em log"""
    m = min(FATORES, key=lambda f: abs(np.log(q * f)))
    return m, float(np.log(q * m))


def candidatos(serie, ret_ibov):
    """linhas de eventos_candidatos.csv de um ativo (indexado por data)"""
    p = serie.fechamento
    razao = p / p.shift()
    suf = (serie.especi.str.split().str[1:].str.join(" ")
           .str.replace(r"\b(NM|N1|N2|MA|MB)\b", "", regex=True).str.strip())
    inicio_flag = suf.str.startswith("E") & (suf != suf.shift())
    neg_med = serie.negocios.rolling(60, min_periods=10).median().shift()
    linhas = []
    for i in range(1, len(p)):
        d = p.index[i]
        r_ib = ret_ibov.get(d, np.nan)
        q = razao.iloc[i] / (1 + (0 if np.isnan(r_ib) else r_ib))
        q_med = p.iloc[i:i + 5].median() / p.iloc[max(0, i - 5):i].median()
        salto = abs(np.log(q)) > LIMIAR_SALTO or (p.iloc[i - 1] < PRECO_CENTAVOS and abs(np.log(q_med)) > LIMIAR_SALTO)
        flag = suf.iloc[i] if inicio_flag.iloc[i] else ""
        if not ((flag and salto) or abs(razao.iloc[i] - 1) > LIMIAR_EXTREMO):
            continue
        linhas.append({"data": d, "ticker": serie.ativo.iloc[i], "ticker_negociado": serie.ticker.iloc[i],
                       "preco_anterior": p.iloc[i - 1], "preco": p.iloc[i], "razao": razao.iloc[i],
                       "ret_ibov": r_ib, "razao_ajustada": q, "razao_mediana_5": q_med,
                       "especi": serie.especi.iloc[i], "flag_ex": flag,
                       "negocios": serie.negocios.iloc[i], "negocios_mediana_60": neg_med.iloc[i]})
    return linhas


def classificar(c):
    """decide se o candidato eh evento e qual fator aplicar antes da data

    fator_preco multiplica todos os precos antes da data (1 = nada)"""
    chave = (c["ticker"], c["data"].date().isoformat())
    flag, q = c["flag_ex"], c["razao_ajustada"]
    if chave in CONFIRMADOS:
        tipo, x, fonte = CONFIRMADOS[chave]
        if tipo == "acoes":
            return "evento em ações", 1 / x, True, fonte
        return "provento extraordinário", (c["preco_anterior"] - x) / c["preco_anterior"], True, fonte
    if not flag:
        return "real (sem marcação ex)", 1.0, False, "movimento de mercado mantido"
    if not re.fullmatch(r"E[BCDGJRSX]+", flag):   # ex: "E" sozinho (OGXP3, 2013), n ta na tabela ESPECI
        return "real (marcação não documentada)", 1.0, False, f"marcação {flag} fora do layout; mantido"
    em_acoes = any(x in flag for x in "BGX")      # bonificacao, grupamento, desdobramento
    if em_acoes:
        # em acao de centavos o tick de 0.01 distorce a razao, testa a mediana tb
        opcoes = [q] + ([c["razao_mediana_5"]] if c["preco_anterior"] < PRECO_CENTAVOS else [])
        m, erro = min((fator_redondo(x) for x in opcoes), key=lambda me: abs(me[1]))
        if abs(erro) <= TOLERANCIA_FATOR:
            return "evento em ações", 1 / m, True, f"marcação {flag} + fator redondo {m:.4g} (erro {erro:+.1%})"
        return "evento em ações (neutralizado)", c["razao"], True, f"marcação {flag}, sem fator redondo"
    if 1 - q > LIMIAR_DISTRIBUICAO:              # cisao, reducao de capital, provento grande
        return "distribuição (neutralizada)", c["razao"], True, f"marcação {flag}, queda ajustada de {1 - q:.0%}"
    return "provento sem ajuste", 1.0, False, f"marcação {flag}; valor do provento indisponível (D24)"


def main():
    longo = pd.read_csv(PASTA / "precos_b3_long.csv", parse_dates=["data"])
    ibov = pd.read_csv(RAIZ / "dados_brutos" / "yahoo" / "ibov.csv", parse_dates=["Date"], index_col="Date").ibov
    ret_ibov = ibov.pct_change()

    cand, series = [], {}
    for ativo, g in longo.groupby("ativo"):
        s = g.sort_values("data").set_index("data")
        assert s.index.is_unique, ativo        # emenda sem sobrepor (ARTR3: OHLB3 ate 21/12/2012)
        series[ativo] = s
        cand += candidatos(s, ret_ibov)
    cand = pd.DataFrame(cand)
    cls = cand.apply(classificar, axis=1, result_type="expand")
    cand[["classificacao", "fator_preco", "aplicado", "evidencia"]] = cls
    cand.to_csv(PASTA / "eventos_candidatos.csv", index=False, float_format="%.6g")

    partes, qualidade = [], []
    cal = ibov.index[(ibov.index >= INICIO) & (ibov.index < FIM)]
    for ativo, s in series.items():
        ev = cand[(cand.ticker == ativo) & cand.aplicado]
        fator = pd.Series(1.0, s.index)
        for e in ev.itertuples():               # ajuste pra tras: precos < data x fator
            fator[s.index < e.data] *= e.fator_preco
        adj = s.fechamento * fator
        partes.append(pd.DataFrame({"data": s.index, "ticker": ativo, "ticker_negociado": s.ticker.values,
                                    "isin": s["isin"].values, "close": s.fechamento.values, "adj_close": adj.values,
                                    "volume": s.qtd.values, "volume_financeiro": s.volume_financeiro.values,
                                    "negocios": s.negocios.values}))
        dias = cal[(cal >= s.index.min()) & (cal <= s.index.max())]
        com_negocio = s.index[(s.index >= INICIO) & (s.index < FIM)]
        classe, descr = FIM_SERIE[ativo]
        qualidade.append({
            "ticker": ativo, "tickers_isin": " -> ".join(dict.fromkeys(s.ticker)),
            "isins": " -> ".join(dict.fromkeys(s["isin"])), "primeiro_pregao": s.index.min().date(),
            "ultimo_pregao": s.index.max().date(), "n_pregoes": len(com_negocio),
            "pct_pregoes_sem_negocio": 1 - len(com_negocio.intersection(dias)) / len(dias) if len(dias) else np.nan,
            "eventos_aplicados": "; ".join(f"{e.data.date()} x{e.fator_preco:.4g} ({e.classificacao})"
                                           for e in ev.itertuples()),
            "n_eventos_aplicados": len(ev), "causa_fim": classe, "detalhe_fim": descr,
            "cobre_janela_2014": bool(s.index.min() <= pd.Timestamp("2011-01-03")
                                      and s.index.max() >= pd.Timestamp("2013-12-30")),
        })

    aj = pd.concat(partes, ignore_index=True)
    aj = aj[(aj.data >= INICIO) & (aj.data < FIM)]
    aj.to_csv(PASTA / "precos_b3_ajustados_long.csv", index=False)
    aj.pivot(index="data", columns="ticker", values="adj_close").sort_index() \
        .to_csv(PASTA / "matriz_adj_close_b3.csv", index_label="Date")
    qual = pd.DataFrame(qualidade)
    qual.to_csv(PASTA / "qualidade_b3.csv", index=False)

    faltam = sorted(set(ALVOS) - set(series))
    meta = {"fonte": "B3 COTAHIST (preço bruto), ajustado por eventos em ações e distribuições grandes",
            "inicio": INICIO, "fim_exclusivo": FIM, "n_ativos": len(series), "nao_encontrados": faltam,
            "proventos_comuns_ajustados": False, "n_candidatos": len(cand),
            "n_eventos_aplicados": int(cand.aplicado.sum())}
    (PASTA / "metadados.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(cand[cand.flag_ex != ""][["data", "ticker", "razao", "razao_ajustada", "flag_ex", "classificacao",
                                    "fator_preco", "evidencia"]].to_string(index=False, float_format="%.4f"))
    print(f"\n{len(cand)} candidatos, {int(cand.aplicado.sum())} aplicados; "
          f"{int((cand.classificacao == 'real (sem marcação ex)').sum())} movimentos reais mantidos")
    print("\n" + qual.drop(columns=["eventos_aplicados"]).to_string(index=False))


if __name__ == "__main__":
    main()
