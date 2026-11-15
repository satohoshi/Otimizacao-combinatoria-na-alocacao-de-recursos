#!/usr/bin/env bash
# demo: validacao, cplex (variancia e cvar) e nsga2
# uso:  bash scripts/demo.sh
# cada passo espera ENTER

cd "$(dirname "$0")/.."                      # vai pra Projeto/

PY=../venv/bin/python
BIN=./build/portfolio_opt
D=data/instancias/2014-01-02_b3/amostral
S=results/comparacao/2014-01-02_b3_amostral/instancia/scenarios.csv

passo() {
    echo
    echo "=================================================================="
    echo "  $1"
    echo "=================================================================="
    read -r -p "  [ENTER para rodar] "
}

passo "0. Compilação e testes C++"
cmake --build build
(cd build && ctest)

passo "1. Validação: CPLEX x enumeração exaustiva (N = 10)"
$PY scripts/validar_enumeracao.py | tail -4

passo "2. CPLEX + variância: N = 87, K = 5, R = 15%, 1% <= w <= 30%"
$BIN --assets $D/assets.csv --cov $D/covariance.csv \
     --K 5 --l 0.01 --u 0.30 --R 0.15 | $PY -m json.tool

passo "3. CPLEX + CVaR 95%: mesma instância, 740 cenários diários"
$BIN --assets $D/assets.csv --cov $D/covariance.csv --cenarios $S \
     --risco cvar --K 5 --l 0.01 --u 0.30 --R 0.15 | $PY -m json.tool

passo "4. NSGA-II + variância: frente inteira de uma vez (amostra de ~10 pontos)"
$BIN --assets $D/assets.csv --cov $D/covariance.csv --cenarios $S \
     --metodo nsga2 --K 5 --l 0.01 --u 0.30 --seed 1 | $PY -c "$(cat <<'EOF'
import json, sys
r = json.load(sys.stdin)
frente = r['frente']
print(f"{len(frente)} carteiras na frente, {r['avaliacoes']} avaliações, {r['tempo_s']:.2f} s")
print(f"{'risco':>10} {'retorno':>9}  ativos")
for c in frente[::max(1, len(frente) // 10)]:
    print(f"{c['risco']:10.5f} {c['retorno']:9.2%}  {', '.join(c['pesos'])}")
EOF
)"

passo "5. Resultados do backtest 2014-2026 (já gerados)"
open ../graficos_resultados/
open ../tabelas_resultados/TABELAS.md

echo
echo "Fim da demonstração."
