//portfolio_opt --assets A.csv --cov C.csv --K 5 [--R 0.10] [--l 0.01] [--u 0.30]
//                [--time-limit 60] [--gap 1e-4] [--threads 0] [--out pesos.csv]
//                [--risco variancia|cvar] [--cenarios S.csv] [--beta 0.95]
//                [--metodo cplex|nsga2] [--pop 100] [--geracoes 500] [--seed 1]
//                [--pc 0.9] [--pm -1] [--eta-c 20] [--eta-m 20]
//
//sem --R faz a carteira de risco minimo
//cvar precisa de --cenarios (S x N, mesma ordem de --assets)
//com --cenarios avalia cada solucao nas duas medidas de risco
//nsga2 ignora --R e devolve a frente toda

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "portfolio/Nsga2.hpp"
#include "portfolio/PortfolioData.hpp"
#include "portfolio/PortfolioOptimizer.hpp"
#include "portfolio/RiskMeasure.hpp"

namespace {

//receber argumentios do terminal
std::map<std::string, std::string> parseArgs(int argc, char** argv) {
    std::map<std::string, std::string> a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        a[k.substr(2)] = argv[++i];
    }
    return a;
}

std::string jsonNum(double x) {
    if (!std::isfinite(x)) return "null";
    std::ostringstream s;
    s << std::setprecision(12) << x;
    return s.str();
}

std::string jsonStr(const std::string& x) { return "\"" + x + "\""; }

}

int main(int argc, char** argv) {
    try {
        //leitura entrada
        auto a = parseArgs(argc, argv);
        auto get = [&](const std::string& k, const std::string& def) {
            return a.count(k) ? a[k] : def;
        };
        //instancia os dados de covariancia e ativos
        auto data = portfolio::PortfolioData::loadFromCsv(get("assets", "data/assets.csv"),
                                                          get("cov", "data/covariance.csv"));
        const int n = static_cast<int>(data.numAssets());

        //insntancia dos parametros, se n teve entrada -> coloca padrao            
        portfolio::OptimizerParams p;
        p.cardinality = std::stoi(get("K", "5"));
        if (a.count("R")) p.targetReturn = std::stod(a["R"]);
        p.minWeight = std::stod(get("l", "0"));
        p.maxWeight = std::stod(get("u", "1"));
        p.timeLimit = std::stod(get("time-limit", "60"));
        p.mipGap = std::stod(get("gap", "1e-4"));
        p.threads = std::stoi(get("threads", "0"));

        const std::string metodo = get("metodo", "cplex");
        const std::string medida = get("risco", "variancia");
        const double beta = std::stod(get("beta", "0.95"));
        std::unique_ptr<portfolio::CVaRRisk> cvar;
        if (a.count("cenarios"))
            cvar = std::make_unique<portfolio::CVaRRisk>(
                portfolio::PortfolioData::loadScenariosCsv(a["cenarios"], data.numAssets()), beta);

        portfolio::VarianceRisk variance(data);
        const portfolio::RiskMeasure& risk =
            medida == "cvar" ? static_cast<const portfolio::RiskMeasure&>(*cvar) : variance;
        const int S = cvar ? static_cast<int>(cvar->numScenarios()) : 0;

        //pesos > 0 e as duas medidas de risco em json
        auto solucaoJson = [&](const portfolio::PortfolioResult& r) {
            std::ostringstream js;
            js << "\"risco\":" << jsonNum(r.risk) << ",\"retorno\":" << jsonNum(r.expectedReturn)
               << ",\"variancia\":" << (r.feasible ? jsonNum(variance.evaluate(r.weights)) : "null")
               << ",\"cvar\":" << (r.feasible && cvar ? jsonNum(cvar->evaluate(r.weights)) : "null")
               << ",\"var\":" << (r.feasible && cvar ? jsonNum(cvar->valueAtRisk(r.weights)) : "null")
               << ",\"pesos\":{";
            bool first = true;
            for (int i = 0; r.feasible && i < n; ++i) {
                if (r.weights[i] <= 1e-9) continue;
                js << (first ? "" : ",") << jsonStr(data.assets()[i].name) << ":" << jsonNum(r.weights[i]);
                first = false;
            }
            js << "}";
            return js.str();
        };
        std::ostringstream cab;
        cab << "\"metodo\":" << jsonStr(metodo) << ",\"medida_risco\":" << jsonStr(risk.name())
            << ",\"beta\":" << (cvar ? jsonNum(beta) : "null") << ",\"S\":" << S << ",\"N\":" << n
            << ",\"K\":" << p.cardinality << ",\"l\":" << jsonNum(p.minWeight)
            << ",\"u\":" << jsonNum(p.maxWeight);

        if (metodo == "nsga2") {
            portfolio::Nsga2Params np;
            np.cardinality = p.cardinality;
            np.minWeight = p.minWeight;
            np.maxWeight = p.maxWeight;
            np.population = std::stoi(get("pop", "100"));
            np.generations = std::stoi(get("geracoes", "500"));
            np.seed = std::stoul(get("seed", "1"));
            np.crossoverProb = std::stod(get("pc", "0.9"));
            np.mutationProb = std::stod(get("pm", "-1"));
            np.etaCrossover = std::stod(get("eta-c", "20"));
            np.etaMutation = std::stod(get("eta-m", "20"));
            np.timeLimit = a.count("time-limit") ? std::stod(a["time-limit"]) : 0.0;
            portfolio::Nsga2 ga(data, risk);
            auto res = ga.run(np);
            std::ostringstream js;
            js << "{\"status\":" << jsonStr(res.front.empty() ? "Infeasible" : "NSGA-II")
               << ",\"viavel\":" << (res.front.empty() ? "false" : "true") << "," << cab.str()
               << ",\"pop\":" << np.population << ",\"geracoes\":" << res.generations
               << ",\"seed\":" << np.seed << ",\"avaliacoes\":" << res.evaluations
               << ",\"tempo_s\":" << jsonNum(res.solveTime) << ",\"frente\":[";
            for (std::size_t k = 0; k < res.front.size(); ++k)
                js << (k ? "," : "") << "{" << solucaoJson(res.front[k]) << "}";
            js << "]}";
            std::cout << js.str() << std::endl;
            return res.front.empty() ? 2 : 0;
        }

        //instancia modelo, risco e opt separados para conseguir modularizar e ter fdiferenres riscos
        portfolio::PortfolioOptimizer opt(data, risk);
        auto r = opt.solve(p);

        if (a.count("out") && r.feasible) {
            std::ofstream f(a["out"]);
            f << "ativo,peso\n" << std::setprecision(12);
            for (int i = 0; i < n; ++i) f << data.assets()[i].name << "," << r.weights[i] << "\n";
        }

        //imprime uma linha json
        std::ostringstream js;
        js << "{\"status\":" << jsonStr(r.status) << ",\"viavel\":" << (r.feasible ? "true" : "false")
           << ",\"otimo\":" << (r.optimal ? "true" : "false") << "," << cab.str()
           << ",\"R\":" << (p.targetReturn ? jsonNum(*p.targetReturn) : "null")
           << ",\"gap\":" << jsonNum(r.gap) << ",\"limitante\":" << jsonNum(r.bestBound)
           << ",\"tempo_s\":" << jsonNum(r.solveTime) << ",\"nos\":" << r.nodes << ","
           << solucaoJson(r) << "}";
        std::cout << js.str() << std::endl;
        return r.feasible ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << "\n";
        return 1;
    }
}
