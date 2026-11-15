//implementacao do nsga
#pragma once

#include <vector>

#include "portfolio/PortfolioData.hpp"
#include "portfolio/PortfolioOptimizer.hpp"
#include "portfolio/RiskMeasure.hpp"

namespace portfolio {

//parametros do nsga2, padrao igual ao artigo do deb (2002)
struct Nsga2Params {
    int cardinality = 5;          //K, igual no cplex
    double minWeight = 0.0;       //l
    double maxWeight = 1.0;       //u
    int population = 100;         //tem q ser par
    int generations = 500;
    double crossoverProb = 0.9;   //prob de crossover
    double mutationProb = -1.0;   //prob de mutacao por gene, < 0 usa 1/(2N)
    double etaCrossover = 20.0;   //eta do sbx
    double etaMutation = 20.0;    //eta da mutacao
    unsigned long seed = 1;
    double timeLimit = 0.0;       //segundos, 0 = sem limite
};

struct Nsga2Result {
    std::vector<PortfolioResult> front;  //nao dominadas, ordenadas por risco
    double solveTime = 0.0;              //segundos
    int generations = 0;
    long evaluations = 0;                //quantas vezes avaliou a fo
};

//nsga2 com 2 objetivos: min risco e max retorno, com exatamente K ativos
//cromossomo = 2N chaves aleatorias em [0,1]: as K maiores das N primeiras escolhem os ativos
//e as N ultimas dao os pesos, depois repara pra ficar em [l,u] e somar 1
//assim toda solucao ja eh viavel, n precisa penalizar
class Nsga2 {
public:
    Nsga2(const PortfolioData& data, const RiskMeasure& risk);
    Nsga2Result run(const Nsga2Params& params) const;

    //transforma as 2N chaves em pesos viaveis
    static std::vector<double> decode(const std::vector<double>& keys, int K, double l, double u);

private:
    const PortfolioData& data_;
    const RiskMeasure& risk_;
};

}
