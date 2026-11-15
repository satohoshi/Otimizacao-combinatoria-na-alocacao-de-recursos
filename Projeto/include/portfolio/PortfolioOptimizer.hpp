//implementacao do modelo cplex
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "portfolio/PortfolioData.hpp"
#include "portfolio/RiskMeasure.hpp"

namespace portfolio {

//input do modelo
struct OptimizerParams {
    int cardinality = 5;                 // K: número EXATO de ativos na carteira
    std::optional<double> targetReturn;  // R: retorno mínimo (vazio = carteira de risco mínimo)
    double minWeight = 0.0;              // l: peso mínimo de cada ativo escolhido
    double maxWeight = 1.0;              // u: peso máximo de cada ativo escolhido
    double timeLimit = 60.0;             // limite de tempo do CPLEX, em segundos
    double mipGap = 1e-4;                // gap relativo de parada do branch-and-bound
    int threads = 0;                     // 0 = CPLEX decide
};

//saida
struct PortfolioResult {
    std::vector<double> weights;   // 0 para os ativos não escolhidos
    double expectedReturn = 0.0;   // μ' w
    double risk = 0.0;             // valor da medida de risco em w
    bool feasible = false;         // há solução inteira viável
    bool optimal = false;          // provada ótima (dentro do gap)
    std::string status;            // status do CPLEX (Optimal, Feasible, Infeasible, ...)
    double gap = 0.0;              // gap relativo final
    double bestBound = 0.0;        // melhor limitante inferior do risco
    double solveTime = 0.0;        // segundos (relógio de parede)
    long nodes = 0;                // nós do branch-and-bound
};

// Seleção de portfólio com restrição de cardinalidade (MIQP):
//
//   min   risco(w)                          (variância: w' Σ w)
//   s.a.  μ' w >= R                          (se R informado)
//         Σ_i w_i = 1
//         Σ_i z_i = K
//         l z_i <= w_i <= u z_i
//         w_i >= 0,  z_i ∈ {0,1}

class PortfolioOptimizer {
public:
    //construtor, armazena basciamente os dados da instancia de portdata e riskmeas
    PortfolioOptimizer(const PortfolioData& data, const RiskMeasure& risk);

    PortfolioResult solve(const OptimizerParams& params) const;

private:
    const PortfolioData& data_;
    const RiskMeasure& risk_;
};

}
