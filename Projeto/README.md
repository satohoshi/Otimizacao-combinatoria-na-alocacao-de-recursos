# Otimização combinatória na alocação de recursos

TCC (MC030 — PFG, Unicamp, 2026) sobre seleção de carteiras com restrição de
cardinalidade: escolher exatamente *K* ativos dentre *N* e definir quanto
alocar em cada um, minimizando o risco para um retorno mínimo. A escolha
discreta dos ativos torna o problema NP-difícil. Ele é resolvido de forma
exata como MIQP no IBM ILOG CPLEX (API C++) e de forma heurística com o
NSGA-II, usando variância ou CVaR como medida de risco.

## Estrutura

```
Projeto/
├── CMakeLists.txt          # build (acha o CPLEX via CPLEX_ROOT_DIR)
├── include/portfolio/      # headers (Asset, PortfolioData, RiskMeasure, PortfolioOptimizer, Nsga2)
├── src/                    # implementação + main.cpp
├── data/                   # instâncias e estimativas por data de decisão
├── scripts/                # coleta, tratamento dos dados, backtest e análises
├── tests/                  # testes em C++ e Python
└── results/                # saídas (não versionadas)
```

## Modelo

```
min   wᵀ Σ w                             (variância da carteira)
s.a.  μᵀ w ≥ R                           (retorno mínimo, opcional)
      Σ w_i = 1
      Σ z_i = K                          (cardinalidade)
      l · z_i ≤ w_i ≤ u · z_i
      w_i ≥ 0,  z_i ∈ {0,1}
```

Sem `R` sai a carteira de variância mínima com `K` ativos. A medida de risco é
uma interface (`RiskMeasure.hpp`), então dá para trocar a variância pelo CVaR
sem mexer no resto do modelo.

## Compilar

Precisa do [IBM ILOG CPLEX Optimization Studio](https://www.ibm.com/products/ilog-cplex-optimization-studio)
(tem licença acadêmica gratuita). Se `CPLEX_ROOT_DIR` não estiver definido, o
CMake procura em `~/Applications/CPLEX_Studio*` e `/Applications/CPLEX_Studio*`.

```bash
export CPLEX_ROOT_DIR="$HOME/Applications/CPLEX_Studio_Community222"   # opcional

mkdir build && cd build
cmake .. -DBUILD_TESTING=ON
cmake --build .
```

Sem o CPLEX o projeto ainda compila (leitura dos dados, testes e NSGA-II), mas
`PortfolioOptimizer::solve()` lança exceção.

## Ambiente Python

```bash
python3 -m venv ../venv
../venv/bin/pip install -r ../requirements.txt
```

## Rodar

Otimizador direto:

```bash
./build/portfolio_opt --assets data/instancias/2014-01-02_b3/amostral/assets.csv \
                      --cov data/instancias/2014-01-02_b3/amostral/covariance.csv \
                      --K 5 --R 0.15 --l 0.01 --u 0.30 [--time-limit 60] [--gap 1e-4] [--out pesos.csv]

# CVaR 95% no CPLEX (scenarios.csv = S x N retornos diários)
./build/portfolio_opt --assets A.csv --cov C.csv --cenarios S.csv --risco cvar --beta 0.95 --K 9 --l 0.01 --u 0.30

# NSGA-II (não precisa do CPLEX), devolve a frente inteira no campo "frente"
./build/portfolio_opt --assets A.csv --cov C.csv --cenarios S.csv --risco cvar --metodo nsga2 \
                      --K 9 --l 0.01 --u 0.30 --pop 100 --geracoes 500 --seed 1
```

A saída é uma linha JSON (status, risco, retorno, gap, tempo, nós e pesos).
`data/assets.csv` e `data/covariance.csv` têm um exemplo com 5 ativos, e
[`docs/INSTANCIAS.md`](docs/INSTANCIAS.md) explica como montar outras instâncias.

Pipeline completo (as bases brutas não estão no repositório, os scripts de
coleta baixam de novo):

```bash
python3 scripts/coletar_yahoo.py       # preços do Yahoo -> dados_brutos/yahoo/
python3 scripts/coletar_cdi.py         # CDI (SGS 12) -> dados_brutos/bcb/
python3 scripts/coletar_cotahist.py    # COTAHIST 2008-2026 -> dados_brutos/b3/zip/
python3 scripts/parse_cotahist.py      # séries da B3 seguidas pelo ISIN
python3 scripts/ajustar_b3.py          # ajuste por eventos corporativos
python3 scripts/consolidar_bases.py    # base única Yahoo + B3
python3 scripts/preparar_dados.py      # μ e Σ em t = 2014-01-02
python3 scripts/validar_enumeracao.py  # CPLEX x enumeração completa (N = 10)
python3 scripts/fronteira.py           # fronteiras por K
python3 scripts/comparar_metodos.py    # {CPLEX, NSGA-II} x {variância, CVaR} numa data
python3 scripts/backtest.py [--medida variancia|cvar] [--metodo cplex|nsga2] [--regra max_sharpe|var_min]
python3 scripts/comparar_grade.py      # comparação da grade completa do backtest
python3 scripts/gerar_tabelas.py       # tabelas em csv/tex/md
```

Testes:

```bash
(cd build && ctest)
python3 tests/test_dados.py && python3 tests/test_backtest.py && python3 tests/test_b3.py && python3 tests/test_metodos.py
```
