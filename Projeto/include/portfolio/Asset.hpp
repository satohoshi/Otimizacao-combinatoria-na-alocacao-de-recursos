#pragma once

#include <string>

namespace portfolio {

//instancia de uma acao
struct Asset {
    std::string name;
    double expectedReturn = 0.0;
};

}
