#include "portfolio/RiskMeasure.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace portfolio {

//variancia = soma soma w_i w_j cov_ij
double VarianceRisk::evaluate(const std::vector<double>& w) const {
    const auto& cov = data_.covariance();
    const std::size_t n = w.size();
    double variancia = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        //so usa os pesos != 0, fica O(K^2) em vez de O(N^2), ajuda mt no nsga
        if (w[i] == 0.0) {
            continue;
        }
        for (std::size_t j = 0; j < n; ++j) {
            if (w[j] == 0.0) {
                continue;
            }
            variancia += cov[i][j] * w[i] * w[j];
        }
    }
    return variancia;
}

//so se tiver o cplex
#ifdef HAVE_CPLEX
IloExpr VarianceRisk::build(IloEnv env, IloModel, const IloNumVarArray& w) const {
    const auto& cov = data_.covariance();
    const int n = static_cast<int>(data_.numAssets());
    IloExpr expr(env);

    //basucamete to colocando a expresao dentro do modelo, ele podeate saber a covariancia q eh calcula antes, mas n tem os pesos
    for (int i = 0; i < n; ++i) {
        //diagonal principal
        expr += cov[i][i] * w[i] * w[i];

        //otimizacao pra n ter q percorrer 2 vezes, dado a simetria
        for (int j = i + 1; j < n; ++j) {
            if (cov[i][j] == 0.0) {
                continue;
            }
            expr += 2.0 * cov[i][j] * w[i] * w[j];
        }
    }
    return expr;
}
#endif

}

//CVARCVARCVARCVARCVARCVARCVARCVARCVARCVARCVARCVARCVARCVARCVARCVARCVARCVAR

namespace portfolio {

//classe cvar de riskmeasure, construtor
CVaRRisk::CVaRRisk(std::vector<std::vector<double>> scenarios, double beta)
    : scen_(std::move(scenarios)), beta_(beta) {
    if (beta_ <= 0.0 || beta_ >= 1.0) {
        throw std::invalid_argument("beta do CVaR deve estar em (0, 1)");
    }
    if (scen_.empty()) {
        throw std::invalid_argument("CVaR sem cenários");
    }
}

//cenario = como o mercado pode se comportar, aqui pegando as piores perdas
//perda no cenario s = -(r_s' w)
std::vector<double> CVaRRisk::losses(const std::vector<double>& w) const { //ref da funcao no hpp
    const std::size_t S = scen_.size();
    const std::size_t n = w.size();
    std::vector<double> perdas(S);

    for (std::size_t s = 0; s < S; ++s) {
        double retorno = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            //so os ativos com peso > 0
            if (w[i] == 0.0) {
                continue;
            }
            retorno += scen_[s][i] * w[i];
        }
        perdas[s] = -retorno;
    }
    return perdas;
}

//var = quantil beta das perdas
double CVaRRisk::valueAtRisk(const std::vector<double>& w) const {
    std::vector<double> perdas = losses(w);
    const double S = static_cast<double>(perdas.size());

    //pega o k-esimo menor, k = ceil(beta*S)
    //o 1e-9 eh pra n arredondar errado e pular pro k seguinte
    const std::size_t k = static_cast<std::size_t>(std::ceil(beta_ * S - 1e-9));

    std::sort(perdas.begin(), perdas.end());
    return perdas[k - 1];
}

//cvar = a + soma max(0, L_s - a) / ((1-b) S)
double CVaRRisk::evaluate(const std::vector<double>& w) const {
    //calcula no alfa otimo (o var)
    const double alpha = valueAtRisk(w);
    const std::vector<double> perdas = losses(w);
    const double S = static_cast<double>(perdas.size());

    double excesso = 0.0;
    for (double perda : perdas) {
        if (perda > alpha) {
            excesso += perda - alpha;
        }
    }
    return alpha + excesso / ((1.0 - beta_) * S);
}

#ifdef HAVE_CPLEX
//coloca a e u_s no modelo e devolve a expressao do cvar
IloExpr CVaRRisk::build(IloEnv env, IloModel model, const IloNumVarArray& w) const {
    const int S = static_cast<int>(scen_.size());
    const int n = static_cast<int>(w.getSize());

    //alfa livre (vira o var no otimo) e u_s >= 0 (quanto passou no cenario s)
    IloNumVar alpha(env, -IloInfinity, IloInfinity, ILOFLOAT, "alpha");
    IloNumVarArray u(env, S, 0.0, IloInfinity, ILOFLOAT);

    for (int s = 0; s < S; ++s) {
        //u_s >= -r_s' w - alfa, so passando pro outro lado
        IloExpr restricao(env);
        restricao += u[s] + alpha;
        for (int i = 0; i < n; ++i) {
            if (scen_[s][i] == 0.0) {
                continue;
            }
            restricao += scen_[s][i] * w[i];
        }
        model.add(restricao >= 0.0);
        restricao.end();
    }

    IloExpr objetivo(env);
    objetivo += alpha + IloSum(u) / ((1.0 - beta_) * S);
    return objetivo;
}
#endif

}
