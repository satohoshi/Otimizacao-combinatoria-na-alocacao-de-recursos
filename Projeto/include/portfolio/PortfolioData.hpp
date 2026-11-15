#pragma once

#include <string>
#include <vector>

#include "portfolio/Asset.hpp"

namespace portfolio {

//instancia dos dados do portfolio, inclui dados de acoes e risco
class PortfolioData {
public:
    //le os ativos e a covariancia de dois csv
    //assets: nome,retorno_esperado | cov: matriz NxN na mesma ordem dos ativos
    static PortfolioData loadFromCsv(const std::string& assetsCsv,
                                      const std::string& covCsv);

    //cenarios pro cvar: matriz S x N sem cabecalho, mesma ordem dos ativos
    static std::vector<std::vector<double>> loadScenariosCsv(const std::string& csv,
                                                             std::size_t numAssets);

    const std::vector<Asset>& assets() const { return assets_; }
    const std::vector<std::vector<double>>& covariance() const { return covariance_; }
    std::size_t numAssets() const { return assets_.size(); }

//acesso a dados
private:
    std::vector<Asset> assets_;
    std::vector<std::vector<double>> covariance_;
};

}