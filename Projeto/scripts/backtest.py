"""backtest com rebalanceamento usando a carteira de maximo sharpe

em cada data t estima mu e sigma so com [t - 3 anos, t), resolve no c++ e pega
a de maior sharpe ex-ante contra o cdi da janela (ou com --regra var_min a de
variancia minima). entre rebalanceamentos os pesos andam com os precos, a ordem
eh executada no fechamento de t

ativo q para de ser negociado eh vendido no ultimo preco e o dinheiro fica no
caixa rendendo cdi ate a proxima decisao. --congelar usa a regra antiga
(posicao parada no ultimo preco, sem render)

uso:
    python3 scripts/backtest.py                               # sigma amostral, max sharpe, com b3
    python3 scripts/backtest.py --sigma lw --regra var_min    # outras combinacoes
    python3 scripts/backtest.py --reusar                      # reaproveita decisoes*.csv
    python3 scripts/backtest.py --sem-b3 --congelar           # base e regra antigas
    python3 scripts/backtest.py --medida cvar                 # cvar 95% no cplex
    python3 scripts/backtest.py --metodo nsga2 [--medida cvar]  # nsga2 no lugar do cplex

frequencias: sem rebalanceamento (compra em 02/01/2014 e segura), anual, semestral,
trimestral, mensal e semanal. decisoes ja resolvidas em decisoes.csv sao
reaproveitadas (--refazer resolve tudo, --reusar n resolve nada)

saidas em results/backtest/<sigma>_<regra>_<b3|semb3>[_congelar]/ (ou --destino):
    decisoes.csv, pesos_decisoes.csv   uma linha por (t, K) e os pesos
    valor_diario.csv                   valor de cada estrategia por pregao
    metricas.csv                       retorno, vol, sharpe, drawdown, turnover
    exposicao_eventos.csv              degraus e precos faltando nas posicoes
    liquidacoes.csv                    vendas forcadas por fim de negociacao
    valor_acumulado, sharpe_por_k, valor_por_frequencia (.png/.pdf)
"""
import argparse
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dados  # noqa: E402
import comparar_metodos  # noqa: E402
import fronteira  # noqa: E402

RAIZ = dados.RAIZ
SAIDA = RAIZ / "results" / "backtest"
SIGMAS = ("amostral", "lw")
REGRAS = ("max_sharpe", "var_min")

# K -> (l, u). K = 3 usa u = 0.40 pq 3 x 0.30 < 1
CARTEIRAS = {3: (0.01, 0.40), 4: (0.01, 0.30), 5: (0.01, 0.30), 9: (0.01, 0.30),
             15: (0.01, 0.30), 20: (0.01, 0.30), 30: (0.01, 0.30)}
# frequencias pelo mes de decisao (primeiro pregao do mes) mais as especiais:
# semanal (primeiro pregao da semana) e sem_rebal (compra no INICIO e segura)
# ORDEM_FREQ eh so a ordem nas tabelas e figuras
FREQUENCIAS = {"anual": [1], "semestral": [1, 7], "trimestral": [1, 4, 7, 10],
               "mensal": list(range(1, 13))}
ORDEM_FREQ = ["sem_rebal", "anual", "semestral", "trimestral", "mensal", "semanal"]
ROTULO_FREQ = {"sem_rebal": "sem rebalanceamento", "anual": "anual", "semestral": "semestral",
               "trimestral": "trimestral", "mensal": "mensal", "semanal": "semanal"}
CUSTOS = [0.0, 0.001]          # fracao do valor negociado
INICIO = "2014-01-02"
V0 = 100_000.0
PONTOS_R = 25                  # grade inicial de R
PONTOS_REFINO = 9              # pontos extras perto do melhor ponto da grade
FOLGA_FIM = 5                  # serie q acaba mais de 5 pregoes antes do fim = deslistada


# decisao
def taxa_livre(cdi, t):
    """cdi medio anualizado na janela [t - 3 anos, t)"""
    return float(dados.janela(cdi, t).mean() * dados.PREGOES_ANO)


def _sharpe(s, rf):
    """sharpe ex-ante, o denominador eh sempre a vol (raiz da variancia) mesmo
    otimizando cvar, pra dar pra comparar"""
    return (s["retorno"] - rf) / np.sqrt(s["variancia"]) if s["viavel"] else -np.inf


def max_sharpe(inst, mu, rf, K, l, u, extra=()):
    """carteira de maior sharpe ex-ante na fronteira com K ativos

    avalia numa grade de PONTOS_R valores de R entre o risco minimo e o retorno max
    e refina perto do melhor. se nenhum ponto passar do cdi fica a de risco minimo
    `extra` troca a medida de risco (--risco cvar)"""
    base = fronteira.resolver(inst, K, l, u, threads=1, extra=extra)
    if not base["viavel"]:
        return None
    sols = [base]
    grade = np.linspace(base["retorno"], fronteira.retorno_maximo(mu, K, l, u) - 1e-9, PONTOS_R)
    sols += [fronteira.resolver(inst, K, l, u, float(R), threads=1, extra=extra) for R in grade[1:]]

    j = int(np.argmax([_sharpe(s, rf) for s in sols]))
    a, b = grade[max(j - 1, 0)], grade[min(j + 1, PONTOS_R - 1)]
    sols += [fronteira.resolver(inst, K, l, u, float(R), threads=1, extra=extra)
             for R in np.linspace(a, b, PONTOS_REFINO + 2)[1:-1]]
    melhor = max(sols, key=lambda s: _sharpe(s, rf))
    regra = "max_sharpe"
    if _sharpe(melhor, rf) <= 0:
        melhor, regra = base, "var_min (Sharpe <= 0)"
    return {"sol": melhor, "regra": regra, "sharpe_esp": _sharpe(melhor, rf),
            "n_resolucoes": len(sols), "todos_otimos": all(s["otimo"] for s in sols if s["viavel"]),
            "gap_max": max(s["gap"] for s in sols if s["viavel"]),
            "tempo_total_s": sum(s["tempo_s"] for s in sols)}


def var_minima(inst, mu, rf, K, l, u, extra=()):
    """carteira de risco minimo com K ativos (sem retorno minimo)"""
    s = fronteira.resolver(inst, K, l, u, threads=1, extra=extra)
    if not s["viavel"]:
        return None
    return {"sol": s, "regra": "var_min", "sharpe_esp": _sharpe(s, rf),
            "n_resolucoes": 1, "todos_otimos": s["otimo"], "gap_max": s["gap"],
            "tempo_total_s": s["tempo_s"]}


NSGA_POP, NSGA_GERACOES, NSGA_SEMENTE = 100, 500, 1


def nsga_decisao(inst, mu, rf, K, l, u, medida="variancia", regra="max_sharpe"):
    """mesma regra mas com o nsga2: uma rodada da a frente toda e pega a de maior
    sharpe (ou a de risco minimo). semente fixa, entao da pra reproduzir"""
    r = comparar_metodos.rodar(inst, K, medida, metodo="nsga2", l=l, u=u, seed=NSGA_SEMENTE,
                               pop=NSGA_POP, geracoes=NSGA_GERACOES)
    if not r["viavel"]:
        return None
    frente = [dict(f, viavel=True, R=None, otimo=False, gap=float("nan"), tempo_s=r["tempo_s"],
                   limitante=float("nan"), nos=0) for f in r["frente"]]   # ja vem ordenado por risco
    melhor, regra_usada = frente[0], "var_min"
    if regra == "max_sharpe":
        melhor, regra_usada = max(frente, key=lambda s: _sharpe(s, rf)), "max_sharpe"
        if _sharpe(melhor, rf) <= 0:
            melhor, regra_usada = frente[0], "var_min (Sharpe <= 0)"
    return {"sol": melhor, "regra": regra_usada, "sharpe_esp": _sharpe(melhor, rf),
            "n_resolucoes": 1, "todos_otimos": False, "gap_max": float("nan"),
            "tempo_total_s": r["tempo_s"]}


def decidir(brutos, cdi, t, ks=CARTEIRAS, pasta=None, sigma="amostral", regra="max_sharpe",
            medida="variancia", metodo="cplex"):
    """pesos de cada K na data t usando so dados < t, devolve (linhas, pesos)

    medida: variancia ou cvar (cenarios = retornos diarios da janela)
    metodo: cplex (exato) ou nsga2, a regra de escolha eh a mesma"""
    t = pd.Timestamp(t)
    est = dados.estimar(brutos, t)
    extra = ["--risco", "cvar", "--cenarios", "{inst}/scenarios.csv", "--beta",
             str(comparar_metodos.BETA)] if medida == "cvar" else []
    with tempfile.TemporaryDirectory() as tmp:
        inst = Path(pasta) if pasta else Path(tmp)
        dados.exportar_instancia(est, inst, usar_lw=(sigma == "lw"))
        mu = est.mu[est.tickers].values
        rf = taxa_livre(cdi, t)
        extra = [x.format(inst=inst) for x in extra]

        def um(K):
            l, u = ks[K]
            if metodo == "nsga2":
                return K, nsga_decisao(inst, mu, rf, K, l, u, medida, regra)
            f = max_sharpe if regra == "max_sharpe" else var_minima
            return K, f(inst, mu, rf, K, l, u, extra=extra)

        with ThreadPoolExecutor(max_workers=len(ks)) as ex:
            resultados = list(ex.map(um, ks))

    linhas, pesos = [], []
    for K, r in resultados:
        l, u = ks[K]
        if r is None:
            linhas.append({"t": t.date(), "K": K, "l": l, "u": u, "N": len(est.tickers), "rf": rf,
                           "regra": "inviavel"})
            continue
        s = r["sol"]
        linhas.append({"t": t.date(), "K": K, "l": l, "u": u, "N": len(est.tickers), "rf": rf,
                       "regra": r["regra"], "R": s["R"], "retorno_esp": s["retorno"],
                       "vol_esp": np.sqrt(s["variancia"]),
                       "sharpe_esp": r["sharpe_esp"],
                       "n_ativos": len(s["pesos"]), "n_resolucoes": r["n_resolucoes"],
                       "todos_otimos": r["todos_otimos"], "gap_max": r["gap_max"],
                       "tempo_total_s": r["tempo_total_s"]})
        if medida == "cvar":     # coluna so nas configs novas
            linhas[-1]["cvar_esp"] = s["cvar"]
        pesos += [{"t": t.date(), "K": K, "ticker": tk, "peso": p} for tk, p in s["pesos"].items()]
    return linhas, pesos


def datas_decisao(calendario, meses, inicio=INICIO):
    """primeiro pregao de cada mes em `meses`, de `inicio` ate o fim"""
    cal = calendario[calendario >= pd.Timestamp(inicio)]
    s = pd.Series(cal, index=cal)
    primeiros = s.groupby([cal.year, cal.month]).first()
    return [d for d in primeiros if d.month in meses]


def datas_frequencia(calendario, freq, inicio=INICIO):
    """datas de decisao de uma frequencia"""
    if freq in FREQUENCIAS:
        return datas_decisao(calendario, FREQUENCIAS[freq], inicio)
    cal = calendario[calendario >= pd.Timestamp(inicio)]
    if freq == "sem_rebal":
        return [cal[0]]
    if freq == "semanal":
        iso = cal.isocalendar()
        return list(pd.Series(cal, index=cal).groupby([iso.year.values, iso.week.values]).first())
    raise ValueError(f"frequência desconhecida: {freq}")


# simulacao
def retornos_carteira(brutos, neutralizar_degrau):
    """retornos diarios pra carregar as posicoes a partir do INICIO

    preco faltando = ultimo preco (sem retorno). com neutralizar_degrau os degraus
    da estimacao tem retorno zerado"""
    p = brutos.precos[brutos.precos.index >= pd.Timestamp(INICIO)]
    ret = dados.retornos_simples(p.ffill()).reindex(p.index).fillna(0.0)
    ret_ibov = dados.retornos_simples(brutos.ibov).reindex(ret.index).fillna(0.0)
    neg = brutos.negocios[brutos.negocios.index >= pd.Timestamp(INICIO)]
    limpo, eventos = dados.limpar_extremos(ret, ret_ibov, neg)
    return (limpo if neutralizar_degrau else ret), p.isna(), eventos


def ultimos_pregoes(brutos, folga=FOLGA_FIM):
    """ultimo pregao com preco e volume > 0 dos ativos q acabam mais de `folga`
    pregoes antes do fim (deslistados)"""
    cal = brutos.precos.index
    negociado = brutos.precos.notna() & (brutos.volume.reindex_like(brutos.precos) > 0)
    fim = {}
    for tk in brutos.precos.columns:
        dias = cal[negociado[tk].values]
        if len(dias) and cal.get_loc(dias[-1]) < len(cal) - 1 - folga:
            fim[tk] = dias[-1]
    return fim


def simular(ret, pesos_por_data, custo, fim=None, caixa=None, registro=None):
    """valor diario e turnover de cada rebalanceamento

    na data d a carteira ja andou com o retorno de d e rebalanceia no fechamento
    custo = `custo` x valor negociado, a compra inicial tambem paga

    fim: ticker -> ultimo pregao. nesse dia a posicao eh vendida no ultimo preco
    (sem custo) e vai pro caixa rendendo `caixa` (ex: cdi) ate o proximo rebal
    cada venda vai pra `registro`. fim=None usa a regra antiga (posicao congelada)"""
    cols = {tk: i for i, tk in enumerate(ret.columns)}
    R = ret.values
    h = np.zeros(len(cols))
    caixa_v = 0.0
    rc = np.zeros(len(ret)) if caixa is None else caixa.reindex(ret.index).fillna(0.0).values
    datas = ret.index
    fim_i = {cols[tk]: datas.get_loc(d) for tk, d in (fim or {}).items() if tk in cols and d in datas}
    valor = np.full(len(ret), np.nan)
    turnover, custos = [], 0.0
    i0 = datas.get_loc(min(pesos_por_data))
    for i in range(i0, len(datas)):
        if i > i0:
            h = h * (1.0 + R[i])
            caixa_v *= 1.0 + rc[i]
        d = datas[i]
        if d in pesos_por_data:
            V = h.sum() + caixa_v if i > i0 else V0
            alvo = np.zeros_like(h)
            for tk, w in pesos_por_data[d].items():
                alvo[cols[tk]] = w * V
            negociado = np.abs(alvo - h).sum()
            c = custo * negociado
            custos += c
            if i > i0:
                turnover.append(negociado / (2 * V))
            h = alvo * (V - c) / V
            caixa_v = 0.0
        for j, ij in fim_i.items():          # depois do rebal, vale tb se comprou no ultimo dia
            if ij == i and h[j] != 0:
                if registro is not None:
                    registro.append({"data": d, "ticker": ret.columns[j], "valor": h[j]})
                caixa_v += h[j]
                h[j] = 0.0
        valor[i] = h.sum() + caixa_v
    return pd.Series(valor[i0:], datas[i0:]), turnover, custos


def metricas(v, fator_cdi, turnover=None, custos=0.0):
    r = v.pct_change().dropna()
    rc = fator_cdi.reindex(v.index).pct_change().dropna()
    ex = r - rc
    anos = (v.index[-1] - v.index[0]).days / 365.25
    acum = v.iloc[-1] / v.iloc[0] - 1
    m = {"inicio": v.index[0].date(), "fim": v.index[-1].date(), "valor_final": v.iloc[-1],
         "retorno_acumulado": acum, "retorno_anual": (1 + acum) ** (1 / anos) - 1,
         "volatilidade_anual": r.std() * np.sqrt(dados.PREGOES_ANO),
         # o proprio cdi tem excesso ~0 (erro de ponto flutuante), sharpe = 0
         "sharpe_cdi": ex.mean() / ex.std() * np.sqrt(dados.PREGOES_ANO) if ex.abs().max() > 1e-12 else 0.0,
         "drawdown_max": (v / v.cummax() - 1).min()}
    if turnover is not None:
        m |= {"n_rebalanceamentos": len(turnover), "turnover_medio": np.mean(turnover) if turnover else 0,
              "turnover_anual": sum(turnover) / anos, "custos_totais": custos}
    return m


def exposicao(pesos, decisoes_datas, ev, ausente, calendario):
    """degraus e pregoes sem preco nos ativos q tavam na carteira"""
    linhas = []
    for (K, freq), grupo in pesos.groupby(["K", "freq"]):
        ds = sorted(decisoes_datas[freq])
        for a, b in zip(ds, ds[1:] + [calendario[-1] + pd.Timedelta(days=1)]):
            held = set(grupo[grupo.t == a].ticker)
            per = (ausente.index > a) & (ausente.index < b)
            for tk in held:
                faltam = int(ausente.loc[per, tk].sum())
                if faltam:
                    linhas.append({"K": K, "freq": freq, "periodo": a.date(), "ticker": tk,
                                   "tipo": "pregoes_sem_preco", "valor": faltam})
            e = ev[(ev.classificacao == "degrau") & ev.ticker.isin(held)]
            e = e[(pd.to_datetime(e.data) > a) & (pd.to_datetime(e.data) < b)]
            linhas += [{"K": K, "freq": freq, "periodo": a.date(), "ticker": x.ticker,
                        "tipo": f"degrau {x.data}", "valor": x.retorno} for x in e.itertuples()]
    return pd.DataFrame(linhas, columns=["K", "freq", "periodo", "ticker", "tipo", "valor"])


def motivos_fim(brutos):
    """ticker -> motivo do fim da serie (qualidade_b3.csv pra b3)"""
    m = dict.fromkeys(brutos.precos.columns, "fim da série no Yahoo")
    q = dados.BRUTOS_B3 / "qualidade_b3.csv"
    if q.exists():
        q = pd.read_csv(q)
        m |= {r.ticker: f"{r.causa_fim}: {r.detalhe_fim}" for r in q.itertuples() if brutos.origem.get(r.ticker) == "b3"}
    return m


# graficos
def graficos(valor, met, destino, rotulo, ks_figura=(3, 9, 20, 30), k_freq=9):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cores = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    tinta, tinta2, muda, grade = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": tinta2,
                         "xtick.color": muda, "ytick.color": muda})

    def estilo(ax):
        ax.grid(color=grade, lw=0.6)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)

    # fig 1: valor da carteira (anual, sem custo) x ibov e cdi, com drawdown
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1]})
    for cor, K in zip(cores, ks_figura):
        s = valor[f"K{K}_anual_c0.00%"]
        ax.plot(s.index, s / 1000, color=cor, lw=1.6, label=f"K = {K}")
        ax2.plot(s.index, (s / s.cummax() - 1) * 100, color=cor, lw=1.2)
    for nome, ls in [("Ibovespa", "-"), ("CDI", "--")]:
        s = valor[nome]
        ax.plot(s.index, s / 1000, color=tinta2, lw=1.4, ls=ls, label=nome)
        ax2.plot(s.index, (s / s.cummax() - 1) * 100, color=tinta2, lw=1.0, ls=ls)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0f}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("Valor (R$ mil, escala log)")
    ax2.set_ylabel("Drawdown (%)")
    ax.set_title(f"(a) Valor da carteira — {rotulo}, rebalanceamento anual, sem custos", fontsize=10,
                 color=tinta, loc="left")
    ax2.set_title("(b) Queda em relação ao pico anterior", fontsize=10, color=tinta, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    estilo(ax)
    estilo(ax2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(destino / f"valor_acumulado.{ext}", dpi=200, facecolor="white")
    plt.close(fig)

    # fig 2: sharpe realizado por K e frequencia (sem e com custo)
    cores_f = dict(zip(ORDEM_FREQ, ["#52514e", "#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#a23fb5"]))
    marcas_f = dict(zip(ORDEM_FREQ, ["x", "o", "s", "D", "^", "v"]))
    freqs = [f for f in ORDEM_FREQ if f in set(met.freq.dropna())]
    fig, eixos = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    base = met[met.tratamento == "degrau_neutralizado"]
    ks = sorted(CARTEIRAS)
    for ax, custo in zip(eixos, CUSTOS):
        for freq in freqs:
            d = base[(base.freq == freq) & (base.custo == custo)].set_index("K").loc[ks]
            ax.plot(range(len(ks)), d.sharpe_cdi, marker=marcas_f[freq], ms=5.5, lw=1.2,
                    color=cores_f[freq], ls="--" if freq == "sem_rebal" else "-", label=ROTULO_FREQ[freq])
        for nome, ls in [("Ibovespa", "-")]:
            ax.axhline(met.loc[met.estrategia == nome, "sharpe_cdi"].iloc[0], color=tinta2, ls=ls,
                       lw=1)
            ax.annotate(nome, (len(ks) - 1, met.loc[met.estrategia == nome, "sharpe_cdi"].iloc[0]),
                        xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8,
                        color=tinta2)
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.set_xticks(range(len(ks)), [str(k) for k in ks])
        ax.set_xlabel("K (nº de ativos)")
        ax.set_title(f"({'ab'[CUSTOS.index(custo)]}) Custo de transação = {custo:.2%}".replace(".", ","), fontsize=10,
                     color=tinta, loc="left")
        estilo(ax)
    eixos[0].set_ylabel("Sharpe realizado contra o CDI")
    eixos[0].legend(frameon=False, fontsize=8, title="Rebalanceamento", title_fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(destino / f"sharpe_por_k.{ext}", dpi=200, facecolor="white")
    plt.close(fig)

    # fig 3: valor com K fixo em cada frequencia, sem e com custo
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, custo in zip(eixos, CUSTOS):
        for freq in freqs:
            s = valor[f"K{k_freq}_{freq}_c{custo:.2%}"]
            ax.plot(s.index, s / 1000, color=cores_f[freq], lw=1.3,
                    ls="--" if freq == "sem_rebal" else "-", label=ROTULO_FREQ[freq])
        for nome, ls in [("Ibovespa", "-"), ("CDI", ":")]:
            s = valor[nome]
            ax.plot(s.index, s / 1000, color="#898781", lw=1.1, ls=ls, label=nome)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0f}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"({'ab'[CUSTOS.index(custo)]}) Custo de transação = {custo:.2%}".replace(".", ","),
                     fontsize=10, color=tinta, loc="left")
        estilo(ax)
    eixos[0].set_ylabel("Valor (R$ mil, escala log)")
    eixos[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig.suptitle(f"Valor da carteira por frequência de rebalanceamento — {rotulo}, K = {k_freq}",
                 fontsize=10, color=tinta, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(destino / f"valor_por_frequencia.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reusar", action="store_true", help="não resolve nada; só lê decisoes*.csv")
    ap.add_argument("--refazer", action="store_true",
                    help="resolve todas as datas de novo (padrão: reaproveita as datas já em decisoes.csv)")
    ap.add_argument("--sigma", choices=SIGMAS, default="amostral")
    ap.add_argument("--regra", choices=REGRAS, default="max_sharpe")
    ap.add_argument("--medida", choices=("variancia", "cvar"), default="variancia",
                    help="medida de risco otimizada (D30); a escolha do ponto usa sempre o Sharpe (D33)")
    ap.add_argument("--metodo", choices=("cplex", "nsga2"), default="cplex", help="solver (D31)")
    ap.add_argument("--freqs", nargs="+", choices=ORDEM_FREQ,
                    help="frequências a simular (padrão: todas; CPLEX com CVaR: sem_rebal, anual, "
                         "semestral e trimestral, por custo)")
    ap.add_argument("--sem-b3", action="store_true", help="só Yahoo (sem as séries do COTAHIST)")
    ap.add_argument("--congelar", action="store_true",
                    help="regra antiga: ativo deslistado fica congelado no último preço")
    ap.add_argument("--destino", help="pasta de saída (padrão: results/backtest/<config>)")
    ap.add_argument("--semente", type=int, help="semente do NSGA-II (padrão 1; D40)")
    args = ap.parse_args()
    if args.semente is not None:
        globals()["NSGA_SEMENTE"] = args.semente
    nome = f"{args.sigma}_{args.regra}_{'semb3' if args.sem_b3 else 'b3'}" + ("_congelar" if args.congelar else "")
    if (args.medida, args.metodo) != ("variancia", "cplex"):
        nome += f"_{args.medida}_{args.metodo}"   # pastas antigas continuam com o mesmo nome
    freqs = args.freqs or (["sem_rebal", "anual", "semestral", "trimestral"]
                           if (args.medida, args.metodo) == ("cvar", "cplex") else ORDEM_FREQ)
    destino = Path(args.destino) if args.destino else SAIDA / nome
    destino.mkdir(parents=True, exist_ok=True)

    brutos = dados.carregar(usar_b3=not args.sem_b3)
    cdi = dados.carregar_cdi()
    cal = brutos.ibov.index
    datas = {f: datas_frequencia(cal, f) for f in ORDEM_FREQ if f in freqs}
    todas = sorted(set().union(*datas.values()))

    # reaproveita as decisoes ja feitas, como t so depende de dados < t
    # adicionar frequencia so resolve as datas novas
    arq_dec, arq_pes = destino / "decisoes.csv", destino / "pesos_decisoes.csv"
    if arq_dec.exists() and not args.refazer:
        dec = pd.read_csv(arq_dec, parse_dates=["t"])
        pes = pd.read_csv(arq_pes, parse_dates=["t"])
    else:
        dec = pd.DataFrame(columns=["t"])
        pes = pd.DataFrame(columns=["t", "K", "ticker", "peso"])
    faltam = [t for t in todas if t not in set(dec.t)]
    if faltam and args.reusar:
        sys.exit(f"--reusar: faltam {len(faltam)} datas em {arq_dec}")
    if faltam:
        print(f"{len(todas) - len(faltam)} datas reaproveitadas; resolvendo {len(faltam)}", flush=True)
        linhas, pesos = [], []
        for n, t in enumerate(faltam, 1):
            li, pe = decidir(brutos, cdi, t, sigma=args.sigma, regra=args.regra,
                             medida=args.medida, metodo=args.metodo)
            linhas += li
            pesos += pe
            print(f"[{n}/{len(faltam)}] {t.date()}: N = {li[0]['N']}, " + ", ".join(
                f"K{x['K']} SR={x.get('sharpe_esp', float('nan')):.2f}" for x in li), flush=True)
            if n % 50 == 0 or n == len(faltam):   # salva parcial pra poder continuar depois
                novo_d, novo_p = pd.DataFrame(linhas), pd.DataFrame(pesos)
                novo_d["t"], novo_p["t"] = pd.to_datetime(novo_d.t), pd.to_datetime(novo_p.t)
                d_all = pd.concat([dec, novo_d], ignore_index=True)
                p_all = pd.concat([pes, novo_p], ignore_index=True)
                d_all["t"], p_all["t"] = pd.to_datetime(d_all.t), pd.to_datetime(p_all.t)   # pandas 3: concat com vazio vira object
                d_all, p_all = d_all.sort_values(["t", "K"]), p_all.sort_values(["t", "K", "ticker"])
                d_all.assign(t=d_all.t.dt.date).to_csv(arq_dec, index=False)
                p_all.assign(t=p_all.t.dt.date).to_csv(arq_pes, index=False)
        dec = pd.read_csv(arq_dec, parse_dates=["t"])
        pes = pd.read_csv(arq_pes, parse_dates=["t"])
    dec = dec[dec.t.isin(todas)]
    pes = pes[pes.t.isin(todas)]

    # benchmarks a partir do fechamento do INICIO
    fator_cdi = (1 + cdi).cumprod().reindex(cal, method="ffill")
    cal_sim = cal[cal >= pd.Timestamp(INICIO)]
    ibov = brutos.ibov.reindex(cal_sim)
    valor = {"Ibovespa": V0 * ibov / ibov.iloc[0],
             "CDI": V0 * fator_cdi.reindex(cal_sim) / fator_cdi.reindex(cal_sim).iloc[0]}
    met = [{"estrategia": n, **metricas(s, fator_cdi)} for n, s in valor.items()]

    ret_bruto, ausente, eventos = retornos_carteira(brutos, neutralizar_degrau=False)
    ret_limpo, _, _ = retornos_carteira(brutos, neutralizar_degrau=True)
    fim = None if args.congelar else ultimos_pregoes(brutos)
    caixa = dados.carregar_cdi()
    liq = []
    pes_freq = []
    for K in CARTEIRAS:
        for freq, ds in datas.items():
            pk = pes[(pes.K == K) & pes.t.isin(ds)]
            pes_freq.append(pk.assign(freq=freq))
            por_data = {d: g.set_index("ticker").peso for d, g in pk.groupby("t")}
            for custo in CUSTOS:
                for trat, ret in [("degrau_neutralizado", ret_limpo), ("bruto", ret_bruto)]:
                    reg = []
                    v, to, cst = simular(ret, por_data, custo, fim, caixa, reg)
                    nome = f"K{K}_{freq}_c{custo:.2%}"
                    liq += [{"estrategia": nome, "K": K, "freq": freq, "custo": custo, "tratamento": trat, **x}
                            for x in reg]
                    if trat == "degrau_neutralizado":
                        valor[nome] = v
                    met.append({"estrategia": nome, "K": K, "freq": freq, "custo": custo,
                                "tratamento": trat, **metricas(v, fator_cdi, to, cst)})

    valor = pd.DataFrame(valor)
    met = pd.DataFrame(met)
    valor.to_csv(destino / "valor_diario.csv", index_label="data")
    met.to_csv(destino / "metricas.csv", index=False)
    exp = exposicao(pd.concat(pes_freq), datas, eventos, ausente, cal)
    exp.to_csv(destino / "exposicao_eventos.csv", index=False)
    liq = pd.DataFrame(liq, columns=["estrategia", "K", "freq", "custo", "tratamento", "data", "ticker", "valor"])
    if len(liq):
        liq["motivo"] = liq.ticker.map(motivos_fim(brutos))
        liq["preco"] = [brutos.fechamento.at[d, tk] for d, tk in zip(liq.data, liq.ticker)]
        liq["data"] = liq.data.dt.date
    liq.to_csv(destino / "liquidacoes.csv", index=False)
    rotulo = ("Σ amostral" if args.sigma == "amostral" else "Σ Ledoit-Wolf") + ", " + \
        ("máximo Sharpe" if args.regra == "max_sharpe" else "risco mínimo") + \
        f", {'CVaR' if args.medida == 'cvar' else 'variância'}, {'NSGA-II' if args.metodo == 'nsga2' else 'CPLEX'}"
    graficos(valor, met, destino, rotulo)

    pd.set_option("display.width", 200)
    cols = ["estrategia", "retorno_anual", "volatilidade_anual", "sharpe_cdi", "drawdown_max",
            "turnover_anual"]
    principal = met[(met.tratamento.isna()) | ((met.tratamento == "degrau_neutralizado")
                                                & (met.freq == "anual"))]
    print(principal[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nSharpe por frequência (sem custo / com custo):")
    b = met[met.tratamento == "degrau_neutralizado"]
    print(b.pivot_table(index="K", columns=["custo", "freq"], values="sharpe_cdi")
          .reindex(columns=ORDEM_FREQ, level=1).to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\nDecisões: {len(dec)}; regras: {dec.regra.value_counts().to_dict()}; "
          f"todas ótimas: {bool(dec.todos_otimos.all())}; gap máx {dec.gap_max.max():.1e}; "
          f"tempo de otimização {dec.tempo_total_s.sum() / 60:.1f} min")
    print(f"Eventos em posições carregadas: {len(exp)} (ver exposicao_eventos.csv)")
    print(f"Liquidações forçadas: {len(liq)} (ver liquidacoes.csv)")


if __name__ == "__main__":
    main()
