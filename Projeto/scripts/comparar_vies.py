"""tamanho do vies de sobrevivencia: backtest com e sem as 26 series da b3

pra cada config {sigma amostral, lw} x {max sharpe, var min}, rebal anual, sem custo
e com degraus neutralizados:
    semb3_congelar  base e regra antigas (so yahoo, deslistado congelado)
    semb3           so yahoo, com liquidacao
    b3              yahoo + b3, com liquidacao   <- cenario novo
    b3_congelar     yahoo + b3, deslistado congelado (so amostral, max sharpe)
o vies eh medido por b3 - semb3 (mesma regra de deslistagem)

sensibilidade: vender no ultimo preco eh otimista quando a negociacao para por
falencia, entao a b3 roda de novo zerando as posicoes em FALENCIAS no ultimo pregao
outra sensibilidade soma nas series da b3 um rendimento hipotetico de DY_HIPOTESE
ao ano pra separar o vies de proventos n ajustados do vies de sobrevivencia

uso:
    python3 scripts/comparar_vies.py
saidas em results/backtest/:
    vies_sobrevivencia.csv, vies_sobrevivencia.md, vies_sobrevivencia_sensibilidade.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest  # noqa: E402
import dados  # noqa: E402
from comparar_backtests import CONFIGS  # noqa: E402

FALENCIAS = ["OGXP3", "MMXM3", "BPHA3"]     # falencia/recuperacao judicial no qualidade_b3.csv
DY_HIPOTESE = 0.05                 # proventos ao ano somados nas series da b3 (so hipotese)

VARIANTES = ["semb3_congelar", "semb3", "b3", "b3_congelar"]
METRICAS = {"sharpe_cdi": ("Sharpe", 1, "{:.2f}"), "retorno_anual": ("Retorno a.a. (%)", 100, "{:.1f}"),
            "volatilidade_anual": ("Volatilidade (%)", 100, "{:.1f}"), "drawdown_max": ("Drawdown máx. (%)", 100, "{:.1f}")}


def pasta(sigma, regra, variante):
    # a base antiga ta sem sufixo, a reproducao exata ta em *_semb3_congelar
    return backtest.SAIDA / (f"{sigma}_{regra}" if variante == "semb3_congelar" else f"{sigma}_{regra}_{variante}")


def carregar():
    partes = []
    for (sigma, regra), nome in CONFIGS.items():
        for var in VARIANTES:
            arq = pasta(sigma, regra, var) / "metricas.csv"
            if not arq.exists():
                continue
            m = pd.read_csv(arq)
            liq = pasta(sigma, regra, var) / "liquidacoes.csv"
            n_liq = pd.read_csv(liq) if liq.exists() else pd.DataFrame(columns=["estrategia", "tratamento"])
            m = m[(m.tratamento == "degrau_neutralizado") & (m.freq == "anual") & (m.custo == 0)]
            n = n_liq[n_liq.tratamento == "degrau_neutralizado"].groupby("estrategia").size()
            partes.append(m.assign(config=nome, sigma=sigma, regra=regra, variante=var,
                                   n_liquidacoes=m.estrategia.map(n).fillna(0).astype(int)))
    return pd.concat(partes, ignore_index=True)


def sensibilidades():
    """metricas da b3 (anual, sem custo, degraus neutralizados) com as mesmas decisoes
    em 2 cenarios: FALENCIAS valendo zero no ultimo pregao e b3 com DY_HIPOTESE de proventos"""
    brutos = dados.carregar()
    cdi = dados.carregar_cdi()
    cal = brutos.ibov.index
    fator_cdi = (1 + cdi).cumprod().reindex(cal, method="ffill")
    ret, ausente, _ = backtest.retornos_carteira(brutos, neutralizar_degrau=True)
    fim = backtest.ultimos_pregoes(brutos)
    zerada = ret.copy()
    for tk in FALENCIAS:
        zerada.at[fim[tk], tk] = -1.0
    com_dy = ret.copy()
    b3 = [tk for tk, o in brutos.origem.items() if o == "b3"]
    com_dy[b3] += (~ausente[b3]).astype(float) * ((1 + DY_HIPOTESE) ** (1 / dados.PREGOES_ANO) - 1)
    anuais = backtest.datas_decisao(cal, backtest.FREQUENCIAS["anual"])
    linhas = []
    for (sigma, regra), nome in CONFIGS.items():
        pes = pd.read_csv(pasta(sigma, regra, "b3") / "pesos_decisoes.csv", parse_dates=["t"])
        for K in backtest.CARTEIRAS:
            pk = pes[(pes.K == K) & pes.t.isin(anuais)]
            por_data = {t: g.set_index("ticker").peso for t, g in pk.groupby("t")}
            for cenario, r in [("falencia_zerada", zerada), ("proventos_hipoteticos", com_dy)]:
                v, to, cst = backtest.simular(r, por_data, 0.0, fim, cdi)
                linhas.append({"config": nome, "K": K, "cenario": cenario,
                               **backtest.metricas(v, fator_cdi, to, cst)})
    return pd.DataFrame(linhas)


def main():
    d = carregar()
    cols = ["config", "variante", "K"] + list(METRICAS) + ["n_liquidacoes"]
    d[cols].sort_values(["config", "K", "variante"]).to_csv(backtest.SAIDA / "vies_sobrevivencia.csv", index=False)

    linhas = ["# Viés de sobrevivência: com e sem as séries da B3",
              "", "Rebalanceamento anual, sem custos, degraus neutralizados, 2014-01-02 a 2026-10-07.",
              "Diferença = (Yahoo + B3) − (só Yahoo), ambos com liquidação no último pregão (D26).", ""]
    for nome in CONFIGS.values():
        x = d[d.config == nome]
        linhas += [f"## {nome}", "",
                   "| K | " + " | ".join(f"{METRICAS[m][0]} sem B3 | com B3 | Δ" for m in METRICAS) + " | liquidações |",
                   "|---|" + "---|---|---|" * len(METRICAS) + "---|"]
        for K in sorted(x.K.unique()):
            a = x[(x.K == K) & (x.variante == "semb3")].iloc[0]
            b = x[(x.K == K) & (x.variante == "b3")].iloc[0]
            celulas = []
            for m, (_, esc, fmt) in METRICAS.items():
                celulas += [fmt.format(a[m] * esc), fmt.format(b[m] * esc), fmt.format((b[m] - a[m]) * esc)]
            linhas.append(f"| {int(K)} | " + " | ".join(celulas).replace(".", ",") + f" | {b.n_liquidacoes} |")
        linhas.append("")

    sens = sensibilidades()
    sens.to_csv(backtest.SAIDA / "vies_sobrevivencia_sensibilidade.csv", index=False)
    linhas += ["## Sensibilidades (mesmas decisões da variante com B3)", "",
               f"Perda total: posições em {', '.join(FALENCIAS)} valem zero no último pregão.",
               f"Proventos: séries da B3 com {DY_HIPOTESE:.0%} a.a. de proventos hipotéticos (D24).", "",
               "| Configuração | K | Sharpe sem B3 | com B3 | perda total | proventos | "
               "Retorno a.a. (%) sem B3 | com B3 | perda total | proventos |",
               "|---|---|---|---|---|---|---|---|---|---|"]
    pv = sens.pivot_table(index=["config", "K"], columns="cenario", values=["sharpe_cdi", "retorno_anual"])
    for nome in CONFIGS.values():
        for K in backtest.CARTEIRAS:
            a = d[(d.config == nome) & (d.K == K) & (d.variante == "semb3")].iloc[0]
            b = d[(d.config == nome) & (d.K == K) & (d.variante == "b3")].iloc[0]
            x = pv.loc[(nome, K)]
            nums = [f"{v:.2f}" for v in (a.sharpe_cdi, b.sharpe_cdi, x[("sharpe_cdi", "falencia_zerada")],
                                         x[("sharpe_cdi", "proventos_hipoteticos")])]
            nums += [f"{v * 100:.1f}" for v in (a.retorno_anual, b.retorno_anual,
                                                x[("retorno_anual", "falencia_zerada")],
                                                x[("retorno_anual", "proventos_hipoteticos")])]
            linhas.append(f"| {nome} | {K} | " + " | ".join(nums).replace(".", ",") + " |")
    linhas.append("")
    (backtest.SAIDA / "vies_sobrevivencia.md").write_text("\n".join(linhas))
    print("\n".join(linhas))

    # efeito so da regra de deslistagem e conferencia da base antiga
    base = d[(d.config == list(CONFIGS.values())[0])]
    pv = base.pivot_table(index="K", columns="variante", values="sharpe_cdi")
    print("\nSharpe, Σ amostral + máx. Sharpe, por variante:\n" + pv.round(3).to_string())


if __name__ == "__main__":
    main()
