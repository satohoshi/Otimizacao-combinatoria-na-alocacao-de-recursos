"""gera as tabelas do relatorio: desempenho, efeitos, melhor cenario, tempo,
qualidade do cplex e sensibilidade

cada tabela sai em 3 formatos em ../tabelas_resultados/:
    csv/<nome>.csv   numeros crus (ponto decimal)
    tex/<nome>.tex   tabular pronto pro input dentro de um table (virgula decimal)
    TABELAS.md       todas as tabelas em markdown com a origem de cada uma

estatisticas: media, mediana e desvio padrao amostral (ddof = 1), em geral sobre os 7 K

uso:  ../venv/bin/python3 scripts/gerar_tabelas.py
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
GRADE = RAIZ / "results" / "grade"
COMP = GRADE / "comparacao"
TEMPO = RAIZ / "results" / "tempo"
SAIDA = RAIZ.parent / "tabelas_resultados"

REGRAS = {"max_sharpe": "Máximo Sharpe", "var_min": "Risco mínimo"}
MET = {"cplex": "CPLEX", "nsga2": "NSGA-II"}
MED = {"variancia": "Variância", "cvar": "CVaR"}
FREQS = ["sem_rebal", "anual", "semestral", "trimestral", "mensal"]
ROT_F = {"sem_rebal": "Sem rebal.", "anual": "Anual", "semestral": "Semestral", "trimestral": "Trimestral",
         "mensal": "Mensal"}
KS = [3, 4, 5, 9, 15, 20, 30]
ORD = [(r, m, e) for r in REGRAS for m in MET for e in MED]


def dp(x):
    return x.std(ddof=1)


def num(x, casas=2, pct=False, sinal=False):
    if pd.isna(x):
        return "—"
    v = x * 100 if pct else x
    s = f"{v:+.{casas}f}" if sinal else f"{v:.{casas}f}"
    return s.replace(".", ",") + ("\\%" if pct else "")


def _tex_seguro(t):
    """troca os caracteres q o inputenc/OT1 do relatorio n aceita"""
    import re
    t = t.replace("−", "$-$").replace("·", "$\\cdot$").replace("≥", "$\\geq$").replace("×", "$\\times$").replace(" > ", " $>$ ")
    return re.sub(r"(?<![$\\w])-(\d)", r"$-$\1", t)     # sinal de negativo


class Tabelas:
    def __init__(self):
        self.md = ["# Tabelas de resultados", "",
                   "Geradas por `Projeto/scripts/gerar_tabelas.py` (D39). Σ amostral, base Yahoo + B3, "
                   "rebalanceamento de 2014-01-02 a 2026-10-07, K ∈ {3, 4, 5, 9, 15, 20, 30}, métricas "
                   "realizadas fora da amostra, sem custos de transação salvo indicação. Dp = desvio-padrão "
                   "amostral. Cada tabela existe em `csv/` (números crus) e `tex/` (para `\\input{}`).", ""]
        self.n = 0
        for p in ("csv", "tex"):
            (SAIDA / p).mkdir(parents=True, exist_ok=True)

    def add(self, nome, titulo, origem, df_csv, cab, linhas, nota=""):
        """cab: cabecalhos, linhas: lista de listas de str ja formatadas (tex)"""
        self.n += 1
        df_csv.to_csv(SAIDA / "csv" / f"{nome}.csv", index=False)
        alin = "l" * sum(1 for c in cab if c.startswith("@")) or "l"
        cab_l = [c.lstrip("@") for c in cab]
        n_txt = sum(1 for c in cab if c.startswith("@"))
        col = "l" * n_txt + "r" * (len(cab) - n_txt)
        tex = [f"% {titulo}", f"% Origem: {origem}. Gerado por Projeto/scripts/gerar_tabelas.py.",
               f"\\begin{{tabular}}{{{col}}}", "  \\hline", "  " + " & ".join(cab_l) + " \\\\", "  \\hline"]
        tex += ["  " + " & ".join(l_) + " \\\\" for l_ in linhas]
        tex = [tex[0], tex[1]] + [_tex_seguro(x) for x in tex[2:]]
        tex += ["  \\hline", "\\end{tabular}"]
        (SAIDA / "tex" / f"{nome}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
        md_l = lambda s: s.replace("\\%", "%").replace("$", "").replace("\\,", " ")   # noqa: E731
        self.md += [f"## T{self.n}. {titulo}", "", f"Arquivos: `csv/{nome}.csv`, `tex/{nome}.tex`. "
                    f"Origem: {origem}.", ""]
        if nota:
            self.md += [nota, ""]
        self.md += ["| " + " | ".join(md_l(c) for c in cab_l) + " |", "|" + "|".join("---" if c.startswith("@") else "--:"
                                                                   for c in cab) + "|"]
        self.md += ["| " + " | ".join(md_l(x) for x in l_) + " |" for l_ in linhas]
        self.md.append("")
        del alin

    def fechar(self):
        (SAIDA / "TABELAS.md").write_text("\n".join(self.md) + "\n", encoding="utf-8")


def rot(r, m, e):
    return REGRAS[r], MET[m], MED[e]


# desempenho
def desempenho(T, met):
    d = met[met.custo == 0]
    # consolidacao por config nas 4 frequencias com rebal x 7 K (28 casos)
    reb = d[d.freq != "sem_rebal"]
    g = reb.groupby(["regra", "metodo", "medida"])
    agg = g.agg(n=("sharpe_cdi", "size"), sharpe_media=("sharpe_cdi", "mean"), sharpe_mediana=("sharpe_cdi", "median"),
                sharpe_dp=("sharpe_cdi", dp), retorno_media=("retorno_anual", "mean"),
                retorno_dp=("retorno_anual", dp), vol_media=("volatilidade_anual", "mean"),
                vol_dp=("volatilidade_anual", dp), dd_media=("drawdown_max", "mean"),
                dd_dp=("drawdown_max", dp)).reindex(ORD).reset_index()
    T.add("consolidacao_configuracoes",
          "Desempenho realizado por configuração — 4 frequências com rebalanceamento × 7 K (28 casos), sem custos",
          "results/grade/<config>/metricas.csv", agg,
          ["@Cenário", "@Método", "@Medida", "Sharpe média", "Sharpe mediana", "Sharpe dp", "Retorno a.a.",
           "Vol. a.a.", "Drawdown máx."],
          [[*rot(r.regra, r.metodo, r.medida), num(r.sharpe_media), num(r.sharpe_mediana), num(r.sharpe_dp),
            num(r.retorno_media, 1, True), num(r.vol_media, 1, True), num(r.dd_media, 1, True)]
           for r in agg.itertuples()],
          "Retorno, volatilidade e drawdown são médias nos 28 casos. Sem rebalanceamento fica de fora (tabela "
          "por frequência abaixo).")

    # por cenario, combinacao x frequencia, estatisticas nos 7 K
    for regra, nome_r in REGRAS.items():
        x = d[d.regra == regra]
        g = x.groupby(["metodo", "medida", "freq"])
        a = g.agg(sharpe_media=("sharpe_cdi", "mean"), sharpe_mediana=("sharpe_cdi", "median"),
                  sharpe_dp=("sharpe_cdi", dp), retorno_media=("retorno_anual", "mean"),
                  retorno_mediana=("retorno_anual", "median"), retorno_dp=("retorno_anual", dp),
                  vol_media=("volatilidade_anual", "mean"), vol_mediana=("volatilidade_anual", "median"),
                  vol_dp=("volatilidade_anual", dp), dd_media=("drawdown_max", "mean"),
                  dd_mediana=("drawdown_max", "median"), dd_dp=("drawdown_max", dp),
                  turnover_media=("turnover_anual", "mean")).reset_index()
        a["o"] = a.freq.map(FREQS.index)
        a = a.set_index(["metodo", "medida"]).loc[[(m, e) for m in MET for e in MED]].reset_index().sort_values(
            ["metodo", "medida", "o"], key=lambda s: s.map({"cplex": 0, "nsga2": 1, "variancia": 0, "cvar": 1})
            if s.name in ("metodo", "medida") else s).drop(columns="o")
        T.add(f"cenario_{regra}", f"{nome_r} — métricas por método, medida e frequência (estatísticas nos 7 K, "
                                  "sem custos)", "results/grade/<config>/metricas.csv", a,
              ["@Método", "@Medida", "@Frequência", "Sharpe média", "Sharpe mediana", "Sharpe dp", "Retorno a.a.",
               "Vol. a.a.", "Drawdown máx.", "Giro a.a."],
              [[MET[r.metodo], MED[r.medida], ROT_F[r.freq], num(r.sharpe_media), num(r.sharpe_mediana),
                num(r.sharpe_dp), num(r.retorno_media, 1, True), num(r.vol_media, 1, True),
                num(r.dd_media, 1, True), num(r.turnover_media)] for r in a.itertuples()],
              "Retorno, volatilidade e drawdown: média nos 7 K (mediana e dp no csv). Giro = turnover anual "
              "médio (1 = carteira inteira trocada uma vez por ano).")

    # efeito da frequencia por cenario (4 combinacoes x 7 K juntas)
    a = d.groupby(["regra", "freq"]).agg(n=("sharpe_cdi", "size"), sharpe_media=("sharpe_cdi", "mean"),
                                         sharpe_mediana=("sharpe_cdi", "median"), sharpe_dp=("sharpe_cdi", dp),
                                         retorno_media=("retorno_anual", "mean"),
                                         vol_media=("volatilidade_anual", "mean"),
                                         dd_media=("drawdown_max", "mean"),
                                         turnover_media=("turnover_anual", "mean")).reset_index()
    a = a.sort_values(["regra", "freq"], key=lambda s: s.map(FREQS.index) if s.name == "freq" else s)
    T.add("frequencia", "Efeito da frequência de rebalanceamento — 2 métodos × 2 medidas × 7 K (28 casos) por "
                        "cenário, sem custos", "results/grade/<config>/metricas.csv", a,
          ["@Cenário", "@Frequência", "Sharpe média", "Sharpe mediana", "Sharpe dp", "Retorno a.a.", "Vol. a.a.",
           "Drawdown máx.", "Giro a.a."],
          [[REGRAS[r.regra], ROT_F[r.freq], num(r.sharpe_media), num(r.sharpe_mediana), num(r.sharpe_dp),
            num(r.retorno_media, 1, True), num(r.vol_media, 1, True), num(r.dd_media, 1, True),
            num(r.turnover_media)] for r in a.itertuples()])

    # efeito do custo de transacao (0.10%) no sharpe medio
    c = met.groupby(["regra", "metodo", "medida", "custo"]).sharpe_cdi.mean().unstack("custo").reindex(ORD)
    c = c.reset_index().rename(columns={0.0: "sharpe_sem_custo", 0.001: "sharpe_custo_0_10"})
    c["diferenca"] = c.sharpe_custo_0_10 - c.sharpe_sem_custo
    T.add("custo_transacao", "Sharpe médio (5 frequências × 7 K) sem custo e com custo de transação de 0,10%",
          "results/grade/<config>/metricas.csv", c,
          ["@Cenário", "@Método", "@Medida", "Sem custo", "Custo 0,10\\%", "Diferença"],
          [[*rot(r.regra, r.metodo, r.medida), num(r.sharpe_sem_custo), num(r.sharpe_custo_0_10),
            num(r.diferenca, 3, sinal=True)] for r in c.itertuples()])

    # referencias
    return d


def referencias(T, bench):
    b = bench.loc[["Ibovespa", "CDI"], ["retorno_anual", "volatilidade_anual", "sharpe_cdi", "drawdown_max"]]
    T.add("referencias", "Referências de mercado no mesmo período (2014-01-02 a 2026-10-07)",
          "results/grade/<config>/metricas.csv", b.reset_index(),
          ["@Referência", "Retorno a.a.", "Vol. a.a.", "Sharpe", "Drawdown máx."],
          [[n, num(r.retorno_anual, 1, True), num(r.volatilidade_anual, 1, True), num(r.sharpe_cdi),
            num(r.drawdown_max, 1, True)] for n, r in b.iterrows()])


def efeitos(T):
    ef = pd.read_csv(COMP / "efeitos_pareados.csv")
    ef["grupo"] = ef.grupo.str.replace("\n", " · ")
    nomes = {"metodo": "CPLEX − NSGA-II", "medida": "CVaR − variância", "regra": "Máx. Sharpe − risco mín."}
    a = ef.groupby(["fator", "grupo"], sort=False).dif.agg(
        n="size", media="mean", mediana="median", dp=dp, minimo="min", maximo="max",
        pos=lambda x: (x > 0).mean()).reset_index()
    a["o"] = a.fator.map({"regra": 0, "metodo": 1, "medida": 2})
    a = a.sort_values(["o", "grupo"]).drop(columns="o")
    trad = lambda g: (g.replace("variancia", "variância").replace("cvar", "CVaR").replace("cplex", "CPLEX")   # noqa
                      .replace("nsga2", "NSGA-II"))
    T.add("efeitos_pareados", "Efeito de cada escolha no Sharpe realizado — diferenças pareadas (mesmo K e "
                              "frequência, os outros fatores fixos), sem custos",
          "results/grade/comparacao/efeitos_pareados.csv", a,
          ["@Comparação", "@Mantido fixo", "Pares", "Média", "Mediana", "Dp", "Mín.", "Máx.", "\\% > 0"],
          [[nomes[r.fator], trad(r.grupo), str(r.n), num(r.media, 3, sinal=True), num(r.mediana, 3, sinal=True),
            num(r.dp, 3), num(r.minimo, 3, sinal=True), num(r.maximo, 3, sinal=True), num(r.pos, 0, True)]
           for r in a.itertuples()],
          "Cada par compara duas configurações que diferem só no fator indicado (35 pares = 5 frequências × "
          "7 K). Média > 0 favorece o primeiro termo.")


def melhores(T, met):
    d = met[met.custo == 0]
    a = d.groupby(["regra", "metodo", "medida", "freq"]).sharpe_cdi.agg(
        media="mean", mediana="median", dp=dp, minimo="min", maximo="max").reset_index()
    a = a.sort_values("media", ascending=False).head(10)
    T.add("melhores_configuracoes", "As 10 melhores configurações (cenário, método, medida, frequência) pelo "
                                    "Sharpe médio nos 7 K, sem custos",
          "results/grade/<config>/metricas.csv", a,
          ["@Cenário", "@Método", "@Medida", "@Frequência", "Média", "Mediana", "Dp", "Mín.", "Máx."],
          [[*rot(r.regra, r.metodo, r.medida), ROT_F[r.freq], num(r.media), num(r.mediana), num(r.dp),
            num(r.minimo), num(r.maximo)] for r in a.itertuples()])
    b = d.sort_values("sharpe_cdi", ascending=False).head(10)
    T.add("melhores_casos", "Os 10 melhores casos individuais (configuração, K, frequência), sem custos",
          "results/grade/<config>/metricas.csv",
          b[["regra", "metodo", "medida", "freq", "K", "sharpe_cdi", "retorno_anual", "volatilidade_anual",
             "drawdown_max"]],
          ["@Cenário", "@Método", "@Medida", "@Frequência", "K", "Sharpe", "Retorno a.a.", "Vol. a.a.",
           "Drawdown máx."],
          [[*rot(r.regra, r.metodo, r.medida), ROT_F[r.freq], str(r.K), num(r.sharpe_cdi),
            num(r.retorno_anual, 1, True), num(r.volatilidade_anual, 1, True), num(r.drawdown_max, 1, True)]
           for r in b.itertuples()])


# tempo e qualidade
def qualidade_cplex(T, dec):
    c = dec[dec.metodo == "cplex"].copy()
    c["nao_otima"] = ~c.todos_otimos.astype(bool)
    a = c.groupby(["regra", "medida"]).agg(decisoes=("t", "size"), resolucoes=("n_resolucoes", "sum"),
                                           nao_otimas=("nao_otima", "sum"), gap_max=("gap_max", "max")).reset_index()
    ks = c[c.nao_otima].groupby(["regra", "medida"]).K.apply(lambda k: ", ".join(map(str, sorted(set(k)))))
    a["K_afetados"] = [ks.get((r.regra, r.medida), "—") for r in a.itertuples()]
    a["pct_otimas"] = 1 - a.nao_otimas / a.decisoes
    a = a.set_index(["regra", "medida"]).reindex([(r, e) for r in REGRAS for e in MED]).reset_index()
    T.add("qualidade_cplex", "Qualidade das soluções do CPLEX nas 1.078 decisões da grade (154 datas × 7 K): "
                             "limite de 60 s e gap de 10⁻⁴ por resolução",
          "results/grade/<config>_cplex/decisoes.csv", a,
          ["@Cenário", "@Medida", "Decisões", "Resoluções", "\\% ótimas", "Não ótimas", "Gap máx.", "@K afetados"],
          [[REGRAS[r.regra], MED[r.medida], str(r.decisoes), str(int(r.resolucoes)), num(r.pct_otimas, 1, True),
            str(r.nao_otimas), num(r.gap_max, 1, True), r.K_afetados] for r in a.itertuples()],
          "Uma decisão é \"não ótima\" se pelo menos uma das suas resoluções parou no limite de tempo com gap "
          "acima de 10⁻⁴. Os tempos da grade não entram aqui (ver benchmark).")


def tempo_grade(T, dec):
    a = dec.groupby(["regra", "metodo", "medida"]).agg(decisoes=("t", "size"),
                                                       total_min=("tempo_total_s", lambda x: x.sum() / 60),
                                                       media=("tempo_total_s", "mean"),
                                                       mediana=("tempo_total_s", "median"),
                                                       dp=("tempo_total_s", dp)).reindex(ORD).reset_index()
    T.add("tempo_grade", "Tempo de otimização registrado na grade (ordem de grandeza: 7 K em paralelo e parte "
                         "medida em execuções anteriores, D33/D37)", "results/grade/<config>/decisoes.csv", a,
          ["@Cenário", "@Método", "@Medida", "Decisões", "Total (min)", "Média (s)", "Mediana (s)", "Dp (s)"],
          [[*rot(r.regra, r.metodo, r.medida), str(r.decisoes), num(r.total_min, 1), num(r.media), num(r.mediana),
            num(r.dp)] for r in a.itertuples()],
          "Use a tabela do benchmark para comparar métodos; esta serve para dizer quanto custou a grade.")


def tempo_benchmark(T):
    arq = TEMPO / "benchmark.csv"
    if not arq.exists():
        T.md += ["## Tempo de processamento (benchmark D37)", "", "**Pendente:** `results/tempo/benchmark.csv` "
                 "ainda não existe. Rode de novo este script quando `scripts/benchmark_tempo.py` terminar.", ""]
        return
    d = pd.read_csv(arq)
    d = d[d.viavel.astype(bool)].copy()
    d["K"] = d.K.astype(int)
    d["por_resol"] = d.tempo_solver_s / d.n_resolucoes
    nd = d.t.nunique()
    a = d.groupby(["regra", "metodo", "medida"]).agg(
        decisoes=("tempo_parede_s", "size"), media=("tempo_parede_s", "mean"), mediana=("tempo_parede_s", "median"),
        dp=("tempo_parede_s", dp), p90=("tempo_parede_s", lambda x: x.quantile(0.9)),
        maximo=("tempo_parede_s", "max"), resolucoes=("n_resolucoes", "mean"),
        por_resol_media=("por_resol", "mean"), por_resol_mediana=("por_resol", "median")).reindex(ORD).reset_index()
    T.add("tempo_benchmark", f"Tempo de processamento por decisão (s) — benchmark isolado, {nd} datas × 7 K, "
                             "CPLEX em 1 thread, NSGA-II com 3 sementes",
          "results/tempo/benchmark.csv", a,
          ["@Cenário", "@Método", "@Medida", "N", "Média", "Mediana", "Dp", "p90", "Máx.", "Resol./decisão",
           "s/resolução"],
          [[*rot(r.regra, r.metodo, r.medida), str(r.decisoes), num(r.media), num(r.mediana), num(r.dp), num(r.p90),
            num(r.maximo), num(r.resolucoes, 0), num(r.por_resol_media, 3)] for r in a.itertuples()],
          "Tempo por decisão = relógio de parede para escolher a carteira de um K numa data (inclui criar o "
          "processo e montar o modelo). O CPLEX com máximo Sharpe resolve o problema ~34 vezes por decisão; o "
          "NSGA-II devolve a frente inteira numa execução. N do NSGA-II = datas × K × 3 sementes.")

    # por K (mediana e dp)
    linhas, rows = [], []
    for (r, m, e) in ORD:
        x = d[(d.regra == r) & (d.metodo == m) & (d.medida == e)].groupby("K").tempo_parede_s
        med, sd = x.median().reindex(KS), x.std(ddof=1).reindex(KS)
        rows.append({"regra": r, "metodo": m, "medida": e, **{f"K{k}_mediana": med[k] for k in KS},
                     **{f"K{k}_dp": sd[k] for k in KS}})
        linhas.append([*rot(r, m, e)] + [num(med[k]) for k in KS])
    T.add("tempo_por_k", "Mediana do tempo por decisão (s) por K — benchmark isolado",
          "results/tempo/benchmark.csv", pd.DataFrame(rows),
          ["@Cenário", "@Método", "@Medida"] + [f"K = {k}" for k in KS], linhas, "Desvio-padrão por K no csv.")

    # razoes pareadas
    c = d[d.metodo == "cplex"].set_index(["t", "regra", "medida", "K"]).tempo_parede_s
    n = d[d.metodo == "nsga2"].groupby(["t", "regra", "medida", "K"]).tempo_parede_s.mean()
    raz = (c / n).dropna().rename("razao").reset_index()
    a = raz.groupby(["regra", "medida"]).razao.agg(pares="size", media="mean", mediana="median", dp=dp,
                                                   minimo="min", maximo="max").reset_index()
    a = a.set_index(["regra", "medida"]).reindex([(r, e) for r in REGRAS for e in MED]).reset_index()
    T.add("razao_cplex_nsga2", "Quantas vezes o CPLEX é mais lento que o NSGA-II — razão pareada (mesma data, K, "
                               "cenário e medida; NSGA-II = média das 3 sementes)", "results/tempo/benchmark.csv", a,
          ["@Cenário", "@Medida", "Pares", "Média", "Mediana", "Dp", "Mín.", "Máx."],
          [[REGRAS[r.regra], MED[r.medida], str(r.pares), num(r.media, 1), num(r.mediana, 1), num(r.dp, 1),
            num(r.minimo, 1), num(r.maximo, 1)] for r in a.itertuples()],
          "Razão < 1 = CPLEX mais rápido naquele par.")

    v = d.groupby(["t", "regra", "metodo", "medida", "K"]).tempo_parede_s.mean().unstack("medida")
    rv = (v["cvar"] / v["variancia"]).rename("razao").reset_index()
    a = rv.groupby(["regra", "metodo"]).razao.agg(pares="size", media="mean", mediana="median", dp=dp).reset_index()
    a = a.set_index(["regra", "metodo"]).reindex([(r, m) for r in REGRAS for m in MET]).reset_index()
    T.add("razao_cvar_variancia", "Custo da medida de risco — razão de tempo CVaR / variância (mesma data, K, "
                                  "cenário e método)", "results/tempo/benchmark.csv", a,
          ["@Cenário", "@Método", "Pares", "Média", "Mediana", "Dp"],
          [[REGRAS[r.regra], MET[r.metodo], str(r.pares), num(r.media, 1), num(r.mediana, 1), num(r.dp, 1)]
           for r in a.itertuples()])

    ns = d[d.metodo == "nsga2"].groupby(["regra", "medida", "t", "K"]).tempo_parede_s.agg(["mean", "std"])
    cv = (ns["std"] / ns["mean"]).rename("cv").reset_index().groupby(["regra", "medida"]).cv.agg(
        media="mean", mediana="median", maximo="max").reset_index()
    T.add("variacao_sementes_nsga2", "Variação do tempo do NSGA-II entre as 3 sementes (coeficiente de variação "
                                     "por data e K)", "results/tempo/benchmark.csv", cv,
          ["@Cenário", "@Medida", "CV médio", "CV mediano", "CV máx."],
          [[REGRAS[r.regra], MED[r.medida], num(r.media, 1, True), num(r.mediana, 1, True), num(r.maximo, 1, True)]
           for r in cv.itertuples()])

    nd_f = {"anual": 13, "semestral": 26, "trimestral": 52, "mensal": 154}
    m = d.groupby(["regra", "metodo", "medida"]).tempo_parede_s.mean().reindex(ORD)
    p = pd.DataFrame([{"regra": r, "metodo": me, "medida": e, **{f: v * n_ * len(KS) / 60 for f, n_ in nd_f.items()}}
                      for (r, me, e), v in m.items()])
    T.add("custo_backtest", "Custo projetado do backtest 2014-2026 em minutos (nº de datas × 7 K × tempo médio por "
                            "decisão do benchmark, execução sequencial)", "results/tempo/benchmark.csv", p,
          ["@Cenário", "@Método", "@Medida", "Anual (13)", "Semestral (26)", "Trimestral (52)", "Mensal (154)"],
          [[*rot(r.regra, r.metodo, r.medida)] + [num(getattr(r, f), 1) for f in nd_f] for r in p.itertuples()])


def sensibilidade(T):
    arq = COMP / "sensibilidade_mmxm3.csv"
    if not arq.exists():
        return
    d = pd.read_csv(arq)
    a = d.groupby(["regra", "metodo", "medida"])[["sharpe", "sharpe_sem", "vol", "vol_sem"]].mean().reindex(
        ORD).reset_index()
    T.add("sensibilidade_mmxm3", "Sensibilidade ao episódio MMXM3: médias (7 K × 5 frequências) com e sem os "
                                 "pregões de 2020-10-01 a 2021-06-30 (D38)",
          "results/grade/comparacao/sensibilidade_mmxm3.csv", a,
          ["@Cenário", "@Método", "@Medida", "Sharpe", "Sharpe sem episódio", "Vol.", "Vol. sem episódio"],
          [[*rot(r.regra, r.metodo, r.medida), num(r.sharpe), num(r.sharpe_sem), num(r.vol, 1, True),
            num(r.vol_sem, 1, True)] for r in a.itertuples()],
          "Sharpe aqui = excesso diário médio sobre o CDI / desvio × √252, diferente do Sharpe oficial; serve "
          "só para comparar com e sem o episódio.")


def testes(T):
    arq = COMP / "testes_significancia.csv"
    if not arq.exists():
        return
    a = pd.read_csv(arq)
    T.add("testes_significancia", "Testes de significância das diferenças pareadas de Sharpe (D41)",
          "results/grade/comparacao/testes_significancia.csv", a,
          ["@Fator", "@Comparação", "@Mantido fixo", "Pares", "Dif. média", "Wilcoxon p", "IC 95\\% bootstrap",
           "Bootstrap p"],
          [[r.fator.capitalize(), r.comparacao, r.fixo, str(r.pares), num(r.dif_media_oficial, 3, sinal=True),
            num(r.wilcoxon_p, 3), f"[{num(r.ic95_inf, 3, sinal=True)}; {num(r.ic95_sup, 3, sinal=True)}]",
            num(r.bootstrap_p, 3)] for r in a.itertuples()],
          "Dif. média = média das diferenças de Sharpe oficial nos pares. Wilcoxon pareado bicaudal (pares não "
          "independentes: p otimista). Bootstrap em blocos móveis de 21 pregões, 2.000 reamostragens, mesma "
          "reamostragem para todas as séries; IC sobre a diferença média de Sharpe diário anualizado.")


def sementes(T):
    arq = RAIZ / "results" / "grade_sementes" / "sementes_resumo.csv"
    if not arq.exists():
        T.md += ["## Variação entre sementes do NSGA-II (D40)", "", "**Pendente:** rode "
                 "`scripts/comparar_sementes.py` quando `results/grade_sementes/rodar_sementes.sh` terminar.", ""]
        return
    a = pd.read_csv(arq)
    cp = pd.read_csv(RAIZ / "results" / "grade_sementes" / "sementes_cplex.csv").set_index(["cfg", "freq"])
    a["o"] = a.freq.map(FREQS.index)
    a = a.sort_values(["cfg", "o"]).drop(columns="o")
    cols = [c for c in a.columns if c.startswith("sharpe_s")]
    rotc = lambda c: ("Máximo Sharpe" if c.startswith("max") else "Risco mínimo",   # noqa: E731
                      "CVaR" if c.endswith("cvar") else "Variância")
    T.add("sementes_nsga2", "Sharpe médio nos 7 K do NSGA-II por semente, e comparação com o CPLEX (sem custos)",
          "results/grade_sementes/sementes_resumo.csv", a,
          ["@Cenário", "@Medida", "@Frequência"] + [f"Semente {c[-1]}" for c in cols] +
          ["Média", "Dp", "CPLEX", "CPLEX − média"],
          [[*rotc(r.cfg), ROT_F[r.freq]] + [num(getattr(r, c)) for c in cols] +
           [num(r.media), num(r.dp, 3), num(cp.loc[(r.cfg, r.freq)].sharpe_cplex),
            num(cp.loc[(r.cfg, r.freq)].dif_cplex_menos_nsga2, 3, sinal=True)] for r in a.itertuples()])


def main():
    if SAIDA.exists():
        shutil.rmtree(SAIDA)
    T = Tabelas()
    met, bench, dec = [], None, []
    for (r, m, e) in ORD:
        p = GRADE / f"{r}_{e}_{m}"
        x = pd.read_csv(p / "metricas.csv")
        bench = x[x.estrategia.isin(["Ibovespa", "CDI"])].set_index("estrategia")
        met.append(x[x.tratamento == "degrau_neutralizado"].assign(regra=r, metodo=m, medida=e))
        dec.append(pd.read_csv(p / "decisoes.csv", parse_dates=["t"]).assign(regra=r, metodo=m, medida=e))
    met = pd.concat(met)
    met = met[met.freq.isin(FREQS)].copy()
    met["K"] = met.K.astype(int)
    dec = pd.concat(dec)
    dec["K"] = dec.K.astype(int)
    # so as 154 datas mensais, o cplex com max sharpe e cvar so tem essas
    # (as outras pastas tem semanais ou extras de rodadas antigas)
    ref = dec[(dec.regra == "max_sharpe") & (dec.metodo == "cplex") & (dec.medida == "cvar")]
    mensais = set(ref.t.unique())
    assert len(mensais) == 154, len(mensais)
    dec = dec[dec.t.isin(mensais)]

    T.md += ["# 1. Consolidação", ""]
    referencias(T, bench)
    desempenho(T, met)
    T.md += ["# 2. Cruzamentos e melhor cenário", ""]
    efeitos(T)
    testes(T)
    sementes(T)
    melhores(T, met)
    T.md += ["# 3. Tempo de processamento e qualidade da solução", ""]
    tempo_benchmark(T)
    qualidade_cplex(T, dec)
    tempo_grade(T, dec)
    T.md += ["# 4. Sensibilidade", ""]
    sensibilidade(T)
    T.fechar()
    print(f"{T.n} tabelas em {SAIDA}")


if __name__ == "__main__":
    main()
