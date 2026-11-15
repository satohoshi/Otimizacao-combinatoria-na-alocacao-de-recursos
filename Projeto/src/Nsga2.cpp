#include "portfolio/Nsga2.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>

namespace portfolio {

//funcoes locais
namespace {

struct Individual {
    std::vector<double> keys;
    std::vector<double> w;
    double risk = 0.0;
    double ret = 0.0;
    int rank = 0;
    double crowd = 0.0;
};

//a domina b se n eh pior em nenhum objetivo e eh melhor em pelo menos um
bool dominates(const Individual& a, const Individual& b) {
    return a.risk <= b.risk && a.ret >= b.ret && (a.risk < b.risk || a.ret > b.ret);
}

//fast non dominated sort do deb, preenche o rank e devolve as frentes
std::vector<std::vector<int>> nonDominatedSort(std::vector<Individual>& pop) {
    const int n = static_cast<int>(pop.size());
    std::vector<std::vector<int>> dominated(n); //vetor de n listas
    std::vector<int> count(n, 0);
    std::vector<std::vector<int>> fronts(1);
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
            if (p == q) continue;
            if (dominates(pop[p], pop[q])) dominated[p].push_back(q);
            else if (dominates(pop[q], pop[p])) ++count[p];
        }
        if (count[p] == 0) {
            pop[p].rank = 0;
            fronts[0].push_back(p);
        }
    }
    for (std::size_t i = 0; !fronts[i].empty(); ++i) {
        std::vector<int> next;
        for (int p : fronts[i])
            for (int q : dominated[p])
                if (--count[q] == 0) {
                    pop[q].rank = static_cast<int>(i) + 1;
                    next.push_back(q);
                }
        fronts.push_back(std::move(next));
    }
    fronts.pop_back();
    return fronts;
}

void crowdingDistance(std::vector<Individual>& pop, const std::vector<int>& front) {
    for (int i : front) pop[i].crowd = 0.0;
    if (front.size() <= 2) {
        for (int i : front) pop[i].crowd = std::numeric_limits<double>::infinity();
        return;
    }
    auto porObjetivo = [&](auto valor) {
        std::vector<int> f = front;
        std::sort(f.begin(), f.end(), [&](int a, int b) { return valor(pop[a]) < valor(pop[b]); });
        const double lo = valor(pop[f.front()]), hi = valor(pop[f.back()]);
        pop[f.front()].crowd = pop[f.back()].crowd = std::numeric_limits<double>::infinity();
        if (hi - lo <= 0.0) return;
        for (std::size_t k = 1; k + 1 < f.size(); ++k)
            pop[f[k]].crowd += (valor(pop[f[k + 1]]) - valor(pop[f[k - 1]])) / (hi - lo);
    };
    porObjetivo([](const Individual& x) { return x.risk; });
    porObjetivo([](const Individual& x) { return x.ret; });
}

//menor rank ganha, se empatar ganha quem tem mais distancia
bool crowdedLess(const Individual& a, const Individual& b) {
    return a.rank < b.rank || (a.rank == b.rank && a.crowd > b.crowd);
}

}

Nsga2::Nsga2(const PortfolioData& data, const RiskMeasure& risk) : data_(data), risk_(risk) {}

std::vector<double> Nsga2::decode(const std::vector<double>& keys, int K, double l, double u) {
    //N primeiras chaves escolhem os ativos e as N ultimas dao os pesos
    //se usasse a mesma chave pros dois os pesos ficavam tudo perto de 1/K
    const int n = static_cast<int>(keys.size() / 2);
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    //pega as K maiores chaves, empate pelo indice
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) {
        return keys[a] > keys[b] || (keys[a] == keys[b] && a < b);
    });
    std::vector<int> livres(idx.begin(), idx.begin() + K);
    auto x = [&](int i) { return keys[n + i] + 1e-12; };
    std::vector<double> w(n, 0.0);
    double fixo = 0.0;   //soma do q ja travou em u
    //reparo: distribui proporcional as chaves, quem passar de u trava em u
    //e o resto eh redistribuido, no maximo K iteracoes
    while (!livres.empty()) {
        double soma = 0.0;
        for (int i : livres) soma += x(i);
        const double resto = 1.0 - fixo - static_cast<double>(livres.size()) * l;
        std::vector<int> ficam;
        bool prendeu = false;
        for (int i : livres) {
            w[i] = l + resto * x(i) / soma;
            if (w[i] > u) {
                w[i] = u;
                fixo += u;
                prendeu = true;
            } else {
                ficam.push_back(i);
            }
        }
        if (!prendeu) break;
        livres = std::move(ficam);
    }
    return w;
}

Nsga2Result Nsga2::run(const Nsga2Params& p) const {
    const int n = static_cast<int>(data_.numAssets());
    const int K = p.cardinality;
    Nsga2Result out;
    if (K < 1 || K > n || K * p.minWeight > 1.0 + 1e-12 || K * p.maxWeight < 1.0 - 1e-12)
        return out;   //inviavel, devolve vazio
    const int popSize = std::max(4, p.population + (p.population % 2));
    const int genes = 2 * n;   //selecao + pesos
    const double pm = p.mutationProb < 0 ? 1.0 / genes : p.mutationProb;
    const auto& assets = data_.assets();

    std::mt19937_64 rng(p.seed);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    auto avaliar = [&](Individual& x) {
        x.w = decode(x.keys, K, p.minWeight, p.maxWeight);
        x.ret = 0.0;
        for (int i = 0; i < n; ++i) x.ret += assets[i].expectedReturn * x.w[i];
        x.risk = risk_.evaluate(x.w);
        ++out.evaluations;
    };

    std::vector<Individual> pop(popSize);
    for (auto& x : pop) {
        x.keys.resize(genes);
        for (auto& k : x.keys) k = U01(rng);
        avaliar(x);
    }
    for (auto& f : nonDominatedSort(pop)) crowdingDistance(pop, f);

    auto torneio = [&]() -> const Individual& {
        const auto& a = pop[rng() % popSize];
        const auto& b = pop[rng() % popSize];
        return crowdedLess(a, b) ? a : b;
    };

    //crossover sbx, gene a gene com prob 0.5
    auto sbx = [&](std::vector<double>& c1, std::vector<double>& c2) {
        for (int i = 0; i < genes; ++i) {
            if (U01(rng) > 0.5 || std::abs(c1[i] - c2[i]) < 1e-14) continue;
            const double y1 = std::min(c1[i], c2[i]), y2 = std::max(c1[i], c2[i]);
            const double r = U01(rng);
            auto filho = [&](double beta) {
                const double alpha = 2.0 - std::pow(beta, -(p.etaCrossover + 1.0));
                const double bq = r <= 1.0 / alpha
                    ? std::pow(r * alpha, 1.0 / (p.etaCrossover + 1.0))
                    : std::pow(1.0 / (2.0 - r * alpha), 1.0 / (p.etaCrossover + 1.0));
                return bq;
            };
            const double d = y2 - y1;
            const double b1 = filho(1.0 + 2.0 * y1 / d);
            const double b2 = filho(1.0 + 2.0 * (1.0 - y2) / d);
            double v1 = std::clamp(0.5 * ((y1 + y2) - b1 * d), 0.0, 1.0);
            double v2 = std::clamp(0.5 * ((y1 + y2) + b2 * d), 0.0, 1.0);
            if (U01(rng) < 0.5) std::swap(v1, v2);
            c1[i] = v1;
            c2[i] = v2;
        }
    };

    //mutacao polinomial
    auto mutar = [&](std::vector<double>& c) {
        for (int i = 0; i < genes; ++i) {
            if (U01(rng) >= pm) continue;
            const double y = c[i], r = U01(rng), e = p.etaMutation + 1.0;
            double dq;
            if (r < 0.5)
                dq = std::pow(2.0 * r + (1.0 - 2.0 * r) * std::pow(1.0 - y, e), 1.0 / e) - 1.0;
            else
                dq = 1.0 - std::pow(2.0 * (1.0 - r) + 2.0 * (r - 0.5) * std::pow(y, e), 1.0 / e);
            c[i] = std::clamp(y + dq, 0.0, 1.0);
        }
    };

    int g = 0;
    for (; g < p.generations; ++g) {
        if (p.timeLimit > 0 && elapsed() >= p.timeLimit) break;
        std::vector<Individual> filhos;
        filhos.reserve(popSize);
        while (static_cast<int>(filhos.size()) < popSize) {
            Individual a, b;
            a.keys = torneio().keys;
            b.keys = torneio().keys;
            if (U01(rng) < p.crossoverProb) sbx(a.keys, b.keys);
            mutar(a.keys);
            mutar(b.keys);
            avaliar(a);
            avaliar(b);
            filhos.push_back(std::move(a));
            filhos.push_back(std::move(b));
        }
        //elitismo, pais e filhos juntos disputam as vagas
        std::vector<Individual> uniao = std::move(pop);
        for (auto& f : filhos) uniao.push_back(std::move(f));
        auto frentes = nonDominatedSort(uniao);
        std::vector<Individual> nova;
        nova.reserve(popSize);
        for (auto& f : frentes) {
            crowdingDistance(uniao, f);
            if (nova.size() + f.size() <= static_cast<std::size_t>(popSize)) {
                for (int i : f) nova.push_back(std::move(uniao[i]));
            } else {
                std::sort(f.begin(), f.end(),
                          [&](int a, int b) { return uniao[a].crowd > uniao[b].crowd; });
                for (std::size_t k = 0; nova.size() < static_cast<std::size_t>(popSize); ++k)
                    nova.push_back(std::move(uniao[f[k]]));
                break;
            }
        }
        pop = std::move(nova);
        //rank e crowding servem pro proximo torneio
        for (auto& f : nonDominatedSort(pop)) crowdingDistance(pop, f);
    }
    out.generations = g;
    out.solveTime = elapsed();

    //frente final: so rank 0, sem repetidos, ordenado por risco
    std::vector<const Individual*> f0;
    for (const auto& x : pop)
        if (x.rank == 0) f0.push_back(&x);
    std::sort(f0.begin(), f0.end(), [](auto a, auto b) {
        return a->risk < b->risk || (a->risk == b->risk && a->ret > b->ret);
    });
    std::set<std::pair<double, double>> vistos;
    for (auto x : f0) {
        if (!vistos.insert({x->risk, x->ret}).second) continue;
        PortfolioResult r;
        r.weights = x->w;
        r.expectedReturn = x->ret;
        r.risk = x->risk;
        r.feasible = true;
        r.optimal = false;
        r.status = "NSGA-II";
        r.solveTime = out.solveTime;
        out.front.push_back(std::move(r));
    }
    return out;
}

}
