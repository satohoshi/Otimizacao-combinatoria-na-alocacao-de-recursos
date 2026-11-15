#pragma once

#include <string>
#include <vector>

#include "portfolio/PortfolioData.hpp"

#ifdef HAVE_CPLEX
#include <ilcplex/ilocplex.h>
#endif

//namespace -> evitarconflito
namespace portfolio {

//classe mae de risco 
class RiskMeasure {
public:
    virtual ~RiskMeasure() = default;
    virtual std::string name() const = 0;
    virtual double evaluate(const std::vector<double>& w) const = 0;

#ifdef HAVE_CPLEX
    //devolve a expressao a minimizar, no cvar ja adiciona as variaveis e restricoes extras no modelo
    virtual IloExpr build(IloEnv env, IloModel model, const IloNumVarArray& w) const = 0;
#endif
};

//variancia: w' cov w
class VarianceRisk : public RiskMeasure {
public:
    explicit VarianceRisk(const PortfolioData& data) : data_(data) {}
    std::string name() const override { return "variancia"; }
    double evaluate(const std::vector<double>& w) const override;
#ifdef HAVE_CPLEX
    IloExpr build(IloEnv env, IloModel model, const IloNumVarArray& w) const override;
#endif

private:
    const PortfolioData& data_;
};

//cvar da perda L = -r_s' w nos S cenarios historicos (rockafellar e uryasev)
//CVaR = min_a a + 1/((1-b) S) * soma max(0, -r_s' w - a)
//no build adiciona a (livre) e u_s >= 0, fica S+1 variaveis e S restricoes
//ta em retorno diario, cvar = 0.02 -> perda media de 2% nos piores dias
class CVaRRisk : public RiskMeasure {
    // ainda precisa computar os diferentes cenarios
public:
    //scenarios: S x N, linha = cenario, coluna = ativo
    CVaRRisk(std::vector<std::vector<double>> scenarios, double beta);
    std::string name() const override { return "cvar"; }
    double evaluate(const std::vector<double>& w) const override;
    //var (alfa otimo), so pra mostrar junto com o cvar
    double valueAtRisk(const std::vector<double>& w) const;
    double beta() const { return beta_; }
    std::size_t numScenarios() const { return scen_.size(); }
#ifdef HAVE_CPLEX

    IloExpr build(IloEnv env, IloModel model, const IloNumVarArray& w) const override;
#endif

//acesso dados
private:
    std::vector<double> losses(const std::vector<double>& w) const;
    std::vector<std::vector<double>> scen_;
    double beta_;
};

}
