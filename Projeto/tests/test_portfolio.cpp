//teste simples, so ve se a leitura dos dados ta certa

#include <cassert>
#include <iostream>

#include "portfolio/PortfolioData.hpp"

int main() {
    auto data = portfolio::PortfolioData::loadFromCsv("data/assets.csv", "data/covariance.csv");

    assert(data.numAssets() == 5);
    assert(data.assets()[0].name == "A");
    assert(data.covariance().size() == data.numAssets());
    assert(data.covariance()[0][0] == 0.040);

    std::cout << "test_portfolio: OK\n";
    return 0;
}
