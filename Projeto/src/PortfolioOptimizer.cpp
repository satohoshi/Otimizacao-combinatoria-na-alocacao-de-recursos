#include "portfolio/PortfolioOptimizer.hpp"

#include <chrono>
#include <sstream>
#include <stdexcept>

#ifdef HAVE_CPLEX
#include <ilcplex/ilocplex.h>
#endif

namespace portfolio {

PortfolioOptimizer::PortfolioOptimizer(const PortfolioData& data, const RiskMeasure& risk)
    : data_(data), risk_(risk) {}

#ifdef HAVE_CPLEX

PortfolioResult PortfolioOptimizer::solve(const OptimizerParams& p) const {
    const auto n = static_cast<int>(data_.numAssets());
    const auto& assets = data_.assets();

    IloEnv env;
    PortfolioResult result;

    try {
        //basicamente um vetor q recebe parametros
        IloModel model(env);

        IloNumVarArray w(env, n, 0.0, p.maxWeight, ILOFLOAT);
        IloBoolVarArray z(env, n);
        //colocando restricoes
        for (int i = 0; i < n; ++i) {
            model.add(w[i] <= p.maxWeight * z[i]);
            model.add(w[i] >= p.minWeight * z[i]);
        }
        //alocacao
        model.add(IloSum(w) == 1.0);
        //cardinalidade
        model.add(IloSum(z) == p.cardinality);

        if (p.targetReturn) { //so se tiver um retorno minimo dentro
            IloExpr ret(env);
            for (int i = 0; i < n; ++i) ret += assets[i].expectedReturn * w[i];
            model.add(ret >= *p.targetReturn);
            ret.end();
        }
        //funcao objetivo
        IloExpr riskExpr = risk_.build(env, model, w);
        model.add(IloMinimize(env, riskExpr));//minimizar risco
        riskExpr.end();

        //instancia do cplex
        IloCplex cplex(model);
        cplex.setOut(env.getNullStream());
        cplex.setWarning(env.getNullStream());
        cplex.setParam(IloCplex::Param::TimeLimit, p.timeLimit);
        cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, p.mipGap); //erro aceitavel
        cplex.setParam(IloCplex::Param::Threads, p.threads); //threads a serem usadas

        //timer, saber se n estourou limite
        const auto t0 = std::chrono::steady_clock::now();
        result.feasible = cplex.solve();
        result.solveTime =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        std::ostringstream st; //acumula texto em uma str
        st << cplex.getStatus(); //acumulando
        result.status = st.str();
        result.nodes = cplex.getNnodes();
        result.optimal = cplex.getStatus() == IloAlgorithm::Optimal;

        if (result.feasible) {
            result.weights.resize(n);
            for (int i = 0; i < n; ++i) {
                result.weights[i] = cplex.getValue(w[i]);
                result.expectedReturn += assets[i].expectedReturn * result.weights[i];
            }
            result.risk = risk_.evaluate(result.weights);
            result.gap = cplex.getMIPRelativeGap();
            result.bestBound = cplex.getBestObjValue();
        }
    } catch (const IloException& e) {
        std::string msg = e.getMessage();
        env.end();
        throw std::runtime_error("Erro no CPLEX: " + msg);
    }

    env.end();
    return result;
}

#else

PortfolioResult PortfolioOptimizer::solve(const OptimizerParams&) const {
    throw std::runtime_error(
        "Build compilado sem suporte a CPLEX (defina CPLEX_ROOT_DIR e recompile).");
}

#endif

}