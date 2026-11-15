"""copia os graficos de results/ pra ../foto/ com nomes q dizem de onde vem

nomes:
    <sigma>_<regra>[_<medida>_<metodo>]_<grafico>.png     backtest 2014-2026
    <sigma>_[<medida>_<metodo>_]fronteira.png              fronteira por K
    <sigma>_metodos_<grafico>.png                          cplex x nsga2 numa data
    <sigma>_combinacoes_<grafico>.png                      as 4 combinacoes no backtest
sem _<medida>_<metodo> = variancia no cplex
so sigma amostral e base yahoo + b3

uso:  python3 scripts/exportar_fotos.py        (copia o q existe e lista o q falta)
"""
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
FOTO = RAIZ.parent / "foto"
R = RAIZ / "results"
SIGMA = "amostral"

# (origem relativa a results/, destino)
COPIAS = []
# backtest: variancia/cplex (padrao) e as combinacoes com cvar
for sufixo, marca in [("", ""), ("_cvar_cplex", "_cvar_cplex"), ("_cvar_nsga2", "_cvar_nsga2")]:
    for regra in ("max_sharpe", "var_min") if sufixo == "" else ("max_sharpe",):
        pasta = R / "backtest" / f"{SIGMA}_{regra}_b3{sufixo}"
        for g in ("valor_acumulado", "sharpe_por_k", "valor_por_frequencia"):
            nome = f"{SIGMA}_{regra}{marca}_{g}.png"
            if sufixo == "" and g != "valor_acumulado":
                nome = f"{SIGMA}_{regra}_{g}.png"
            COPIAS.append((pasta / f"{g}.png", nome))
COPIAS.append((R / "fronteira" / f"2014-01-02_b3_{SIGMA}" / "fronteira.png", f"{SIGMA}_fronteira.png"))
# comparacao numa data (cplex x nsga2, variancia e cvar)
cm = R / "comparacao" / f"2014-01-02_b3_{SIGMA}"
for g in ("fronteiras", "espacos_cruzados", "tempo_qualidade", "ativos_K5", "ativos_K9", "ativos_K20"):
    COPIAS.append((cm / f"{g}.png", f"{SIGMA}_metodos_{g}.png"))
# as 4 combinacoes no backtest
cb = R / "backtest" / f"combinacoes_{SIGMA}_b3"
for g in ("sharpe_vol_dd", "valor_acumulado", "tempo_otimizacao"):
    COPIAS.append((cb / f"{g}.png", f"{SIGMA}_combinacoes_{g}.png"))
for K in (3, 4, 5, 9, 15, 20, 30):
    COPIAS.append((cb / f"ativos_K{K}.png", f"{SIGMA}_combinacoes_ativos_K{K}.png"))


def main():
    FOTO.mkdir(exist_ok=True)
    feitas, faltam = 0, []
    for src, nome in COPIAS:
        if src.exists():
            shutil.copy2(src, FOTO / nome)
            feitas += 1
        else:
            faltam.append(src.relative_to(RAIZ))
    print(f"{feitas} fotos copiadas para {FOTO}")
    for f in faltam:
        print(f"  ausente: {f}")


if __name__ == "__main__":
    main()
