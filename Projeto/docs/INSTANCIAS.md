# Como criar uma instância do problema

Uma **instância** do problema de otimização de portfólio é composta por dois
arquivos de entrada (`data/assets.csv` e `data/covariance.csv`) mais os
parâmetros do modelo (`OptimizerParams`, preenchidos pela CLI `portfolio_opt`:
`--K`, `--R`, `--l`, `--u`, `--time-limit`, `--gap`, `--threads`). Este
documento explica como montar cada parte. As instâncias reais usadas no
projeto são geradas por `scripts/preparar_dados.py` em
`data/instancias/<t>_<b3|semb3>/{amostral,lw}/`.

## 1. O que define uma instância

| Elemento | Onde entra | Descrição |
|---|---|---|
| N ativos | `assets.csv` | quantidade de ativos candidatos |
| Retorno esperado `r_i` | `assets.csv` | retorno médio esperado de cada ativo |
| Matriz de covariância `Cov` (N×N) | `covariance.csv` | correlação/risco conjunto entre os ativos |
| Cardinalidade `K` | `OptimizerParams::cardinality` (`--K`, padrão 5) | nº **exato** de ativos na carteira (`Σ z_i = K`) |
| Retorno mínimo `R` | `OptimizerParams::targetReturn` (`--R`, opcional) | restrição `μᵀw ≥ R`; sem ela, variância mínima |
| Peso mín./máx. por ativo | `OptimizerParams::minWeight` / `maxWeight` (`--l`, `--u`) | limites de alocação por ativo escolhido |

O orçamento é fixo em `Σ w_i = 1` e o objetivo é sempre minimizar `wᵀΣw`;
não há parâmetro de aversão a risco. A fronteira é traçada variando `R`.

## 2. Formato de `data/assets.csv`

Uma linha por ativo, sem cabeçalho, formato `nome,retorno_esperado`:

```
A,0.10
B,0.12
C,0.08
```

- `nome`: string sem vírgulas (é só um identificador, não precisa ser um ticker real).
- `retorno_esperado`: número decimal (ex.: `0.10` = 10% no período considerado).
- A **ordem das linhas define a ordem dos índices** usada em `covariance.csv` — o ativo da linha `i` aqui é a linha/coluna `i` na matriz de covariância.

## 3. Formato de `data/covariance.csv`

Matriz N×N, sem cabeçalho, separada por vírgulas, na mesma ordem de `assets.csv`:

```
0.040,0.010,0.005
0.010,0.030,0.004
0.005,0.004,0.020
```

Requisitos obrigatórios (não são validados pelo código — verifique antes de usar):

- **Quadrada**: N linhas × N colunas, N = número de ativos em `assets.csv`.
- **Simétrica**: `Cov[i][j] == Cov[j][i]`.
- **Semidefinida positiva**: garante que a variância `wᵀ·Cov·w` nunca seja negativa. Toda matriz de covariância real (calculada a partir de dados) satisfaz isso automaticamente.
- **Diagonal = variância individual** de cada ativo (`Cov[i][i]`), sempre ≥ 0.

## 4. Montando uma instância a partir de dados reais

Se você tem uma série histórica de retornos (ex.: retornos diários/mensais de N ativos), calcule `r_i` e `Cov` com `pandas`/`numpy`:

```python
import pandas as pd

# df: linhas = períodos, colunas = ativos, valores = retornos percentuais
df = pd.read_csv("retornos_historicos.csv", index_col=0)

expected_returns = df.mean()          # r_i
cov_matrix = df.cov()                 # Cov (já simétrica e PSD)

# Exporta no formato esperado pelo projeto
expected_returns.to_csv("data/assets.csv", header=False)
cov_matrix.to_csv("data/covariance.csv", header=False, index=False)
```

Fontes comuns de dados históricos: Yahoo Finance (`yfinance`), B3/CVM (ativos
brasileiros), bases acadêmicas fornecidas pelo orientador.

> Atenção: `to_csv` do pandas para `expected_returns` inclui o nome do ativo
> como índice — confira se o separador e a ausência de aspas ficaram
> corretos antes de rodar o solver.

## 5. Gerando uma instância sintética (para testar escalabilidade)

Para estudar como o tempo de solução do CPLEX cresce com N (relevante para
discutir a NP-dificuldade no TCC), gere instâncias aleatórias controladas:

```python
import numpy as np

rng = np.random.default_rng(seed=42)
n = 50  # tamanho da instância

returns = rng.uniform(0.02, 0.20, size=n)

# Gera uma matriz de covariância válida (simétrica e PSD) a partir de uma
# matriz aleatória A: Cov = A @ A.T é sempre PSD.
A = rng.normal(scale=0.1, size=(n, n))
cov = A @ A.T

with open("data/assets.csv", "w") as f:
    for i, r in enumerate(returns):
        f.write(f"asset_{i},{r:.6f}\n")

np.savetxt("data/covariance.csv", cov, delimiter=",", fmt="%.6f")
```

Sugestão: salvar essas instâncias em `data/instances/<tamanho>/` (crie a
subpasta) para manter um conjunto de benchmarks reprodutível ao longo do TCC.

## 6. Escolhendo os parâmetros do modelo

- **`cardinality` (K)**: comece pequeno (3–10) para instâncias de teste; aumente para observar o impacto no tempo de solução.
- **`targetReturn` (R)**: omita para a carteira de variância mínima; varie entre esse retorno e o retorno máximo viável para traçar a fronteira eficiente (risco × retorno), como em `scripts/fronteira.py`.
- **`minWeight` / `maxWeight`**: definem diversificação mínima/máxima por ativo escolhido (ex.: `minWeight=0.05, maxWeight=0.40` evita concentração excessiva em um único ativo).

## 7. Checklist antes de rodar

- [ ] `assets.csv` tem N linhas, sem cabeçalho, sem linhas em branco no meio.
- [ ] `covariance.csv` tem exatamente N linhas e N colunas por linha.
- [ ] A ordem dos ativos é a mesma nos dois arquivos.
- [ ] A matriz de covariância é simétrica (`Cov[i][j] == Cov[j][i]`).
- [ ] `cardinality ≤ N` (senão o problema é inviável, pois K é exato).
- [ ] `K · minWeight ≤ 1 ≤ K · maxWeight` (senão o problema é inviável).
