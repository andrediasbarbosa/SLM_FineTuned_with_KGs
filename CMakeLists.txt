cmake_minimum_required(VERSION 3.20)
project(slmkg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG 0.8.0
    GIT_SHALLOW TRUE
)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)

find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(POPPLER_CPP REQUIRED poppler-cpp)

set(SLMKG_INCLUDES src ${POPPLER_CPP_INCLUDE_DIRS})
set(SLMKG_LIBS
    nlohmann_json::nlohmann_json
    yaml-cpp::yaml-cpp
    OpenSSL::Crypto
    ${POPPLER_CPP_LIBRARIES}
)
set(SLMKG_LIBS_FULL
    nlohmann_json::nlohmann_json
    yaml-cpp::yaml-cpp
    OpenSSL::Crypto
    OpenSSL::SSL
    CURL::libcurl
    ${POPPLER_CPP_LIBRARIES}
)

add_executable(slmkg src/cli/main.cpp)
target_include_directories(slmkg PRIVATE ${SLMKG_INCLUDES})
target_link_libraries(slmkg PRIVATE ${SLMKG_LIBS_FULL})

enable_testing()
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

add_executable(test_corpus tests/test_corpus.cpp)
target_include_directories(test_corpus PRIVATE ${SLMKG_INCLUDES})
target_link_libraries(test_corpus PRIVATE ${SLMKG_LIBS} Catch2::Catch2WithMain)
target_compile_definitions(test_corpus PRIVATE SLMKG_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
catch_discover_tests(test_corpus PROPERTIES LABELS "unit")

add_executable(test_prompts tests/test_prompts.cpp)
target_include_directories(test_prompts PRIVATE ${SLMKG_INCLUDES})
target_link_libraries(test_prompts PRIVATE ${SLMKG_LIBS} Catch2::Catch2WithMain)
target_compile_definitions(test_prompts PRIVATE SLMKG_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
catch_discover_tests(test_prompts PROPERTIES LABELS "unit")

add_executable(test_datasets tests/test_datasets.cpp)
target_include_directories(test_datasets PRIVATE ${SLMKG_INCLUDES})
target_link_libraries(test_datasets PRIVATE ${SLMKG_LIBS} Catch2::Catch2WithMain)
target_compile_definitions(test_datasets PRIVATE SLMKG_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
catch_discover_tests(test_datasets PROPERTIES LABELS "unit")

add_executable(test_kg tests/test_kg.cpp)
target_include_directories(test_kg PRIVATE ${SLMKG_INCLUDES})
target_link_libraries(test_kg PRIVATE ${SLMKG_LIBS} Catch2::Catch2WithMain)
target_compile_definitions(test_kg PRIVATE SLMKG_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
catch_discover_tests(test_kg PROPERTIES LABELS "unit")

add_executable(test_extraction tests/test_extraction.cpp)
target_include_directories(test_extraction PRIVATE ${SLMKG_INCLUDES})
target_link_libraries(test_extraction PRIVATE ${SLMKG_LIBS_FULL} Catch2::Catch2WithMain)
target_compile_definitions(test_extraction PRIVATE SLMKG_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
catch_discover_tests(test_extraction PROPERTIES LABELS "unit")
