#include "portfolio/PortfolioData.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace portfolio {

namespace {

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

}

//construtor para covariancia
PortfolioData PortfolioData::loadFromCsv(const std::string& assetsCsv,
                                          const std::string& covCsv) {
    PortfolioData data;
    //carrega ativos
    std::ifstream assetsFile(assetsCsv);
    if (!assetsFile) {
        throw std::runtime_error("Não foi possível abrir o arquivo de ativos: " + assetsCsv);
    }

    //para cada ativo na lista, declara ele e inclui
    std::string line;
    while (std::getline(assetsFile, line)) {
        if (line.empty()) continue;
        auto fields = splitCsvLine(line);
        if (fields.size() != 2) {
            throw std::runtime_error("erro");
        }
        Asset asset;
        asset.name = fields[0];
        asset.expectedReturn = std::stod(fields[1]);
        data.assets_.push_back(asset);
    }

    std::ifstream covFile(covCsv);
    if (!covFile) {
        throw std::runtime_error("Não foi possível abrir o arquivo de covariância: " + covCsv);
    }

    while (std::getline(covFile, line)) {
        if (line.empty()) continue;
        auto fields = splitCsvLine(line);
        std::vector<double> row;
        //alocar espaco antes -> evitar ter q fzr no loop
        row.reserve(fields.size());
        for (const auto& f : fields) {
            row.push_back(std::stod(f));
        }
        data.covariance_.push_back(std::move(row));
    }

    if (data.covariance_.size() != data.assets_.size()) {
        throw std::runtime_error("Matriz de covariância não é NxN compatível com o número de ativos.");
    }
    for (const auto& row : data.covariance_) {
        if (row.size() != data.assets_.size()) {
            throw std::runtime_error("Matriz de covariância não é quadrada.");
        }
    }

    return data;
}

//construtor para cvar
std::vector<std::vector<double>> PortfolioData::loadScenariosCsv(const std::string& csv,
                                                                 std::size_t numAssets) {
    std::ifstream f(csv);
    if (!f) throw std::runtime_error("Não foi possível abrir o arquivo de cenários: " + csv);
    std::vector<std::vector<double>> scen;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto fields = splitCsvLine(line);
        if (fields.size() != numAssets)
            throw std::runtime_error("Cenário com " + std::to_string(fields.size()) +
                                     " colunas; esperado " + std::to_string(numAssets) + ".");
        std::vector<double> row;
        row.reserve(fields.size());
        for (const auto& x : fields) row.push_back(std::stod(x));
        scen.push_back(std::move(row));
    }
    if (scen.empty()) throw std::runtime_error("Arquivo de cenários vazio: " + csv);
    return scen;
}

}
