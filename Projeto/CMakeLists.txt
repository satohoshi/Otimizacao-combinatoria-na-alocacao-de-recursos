cmake_minimum_required(VERSION 3.15)
project(portfolio_optimization CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# cplex
# defina CPLEX_ROOT_DIR com o caminho da instalacao, ex:
#   export CPLEX_ROOT_DIR="/opt/ibm/ILOG/CPLEX_Studio2211"
# ou passa -DCPLEX_ROOT_DIR=... no cmake
if(NOT DEFINED CPLEX_ROOT_DIR)
  set(CPLEX_ROOT_DIR $ENV{CPLEX_ROOT_DIR})
endif()
# se n tiver a variavel tenta o caminho padrao no mac
if(NOT CPLEX_ROOT_DIR)
  file(GLOB _cplex_candidatos "$ENV{HOME}/Applications/CPLEX_Studio*" "/Applications/CPLEX_Studio*")
  if(_cplex_candidatos)
    list(GET _cplex_candidatos 0 CPLEX_ROOT_DIR)
  endif()
endif()

if(CPLEX_ROOT_DIR)
  message(STATUS "Usando CPLEX_ROOT_DIR = ${CPLEX_ROOT_DIR}")

  # acha a pasta da plataforma sozinho
  file(GLOB CPLEX_LIB_PLATFORM_DIRS "${CPLEX_ROOT_DIR}/cplex/lib/*")
  file(GLOB CONCERT_LIB_PLATFORM_DIRS "${CPLEX_ROOT_DIR}/concert/lib/*")

  set(CPLEX_INCLUDE_DIR "${CPLEX_ROOT_DIR}/cplex/include")
  set(CONCERT_INCLUDE_DIR "${CPLEX_ROOT_DIR}/concert/include")

  find_library(CPLEX_LIBRARY
    NAMES cplex
    PATHS ${CPLEX_LIB_PLATFORM_DIRS}
    PATH_SUFFIXES static_pic
  )
  find_library(ILOCPLEX_LIBRARY
    NAMES ilocplex
    PATHS ${CPLEX_LIB_PLATFORM_DIRS}
    PATH_SUFFIXES static_pic
  )
  find_library(CONCERT_LIBRARY
    NAMES concert
    PATHS ${CONCERT_LIB_PLATFORM_DIRS}
    PATH_SUFFIXES static_pic
  )

  set(CPLEX_FOUND TRUE)
  if(NOT CPLEX_LIBRARY OR NOT ILOCPLEX_LIBRARY OR NOT CONCERT_LIBRARY)
    set(CPLEX_FOUND FALSE)
    message(WARNING "Não foi possível localizar as libs do CPLEX/Concert dentro de ${CPLEX_ROOT_DIR}. Compilando sem suporte a CPLEX (apenas leitura de dados).")
  endif()
else()
  set(CPLEX_FOUND FALSE)
  message(WARNING "CPLEX_ROOT_DIR não definido. Compilando sem suporte a CPLEX (apenas leitura de dados). Defina CPLEX_ROOT_DIR para habilitar o solver.")
endif()

# biblioteca
add_library(portfolio_core
  src/PortfolioData.cpp
  src/PortfolioOptimizer.cpp
  src/RiskMeasure.cpp
  src/Nsga2.cpp
)
target_include_directories(portfolio_core PUBLIC include)

if(CPLEX_FOUND)
  target_compile_definitions(portfolio_core PUBLIC HAVE_CPLEX)
  target_include_directories(portfolio_core PUBLIC ${CPLEX_INCLUDE_DIR} ${CONCERT_INCLUDE_DIR})
  target_compile_definitions(portfolio_core PUBLIC IL_STD)
  find_package(Threads REQUIRED)
  target_link_libraries(portfolio_core PUBLIC
    ${ILOCPLEX_LIBRARY} ${CPLEX_LIBRARY} ${CONCERT_LIBRARY}
    Threads::Threads ${CMAKE_DL_LIBS}
  )
  if(APPLE)
    # mesmos frameworks do exemplo da ibm, o accelerate da o blas
    target_link_libraries(portfolio_core PUBLIC
      "-framework CoreFoundation" "-framework IOKit" "-framework Accelerate")
  endif()
endif()

# executavel
add_executable(portfolio_opt src/main.cpp)
target_link_libraries(portfolio_opt PRIVATE portfolio_core)

# testes (-DBUILD_TESTING=ON)
option(BUILD_TESTING "Compilar testes" OFF)
if(BUILD_TESTING)
  enable_testing()
  add_executable(test_portfolio tests/test_portfolio.cpp)
  target_link_libraries(test_portfolio PRIVATE portfolio_core)
  add_test(NAME test_portfolio COMMAND test_portfolio WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
