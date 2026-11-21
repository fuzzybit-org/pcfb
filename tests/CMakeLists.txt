set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
set(CATCH_INSTALL_EXTRAS ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)

add_executable(pcfb_tests
    test_threshold.cpp
    test_builder.cpp
    test_validate.cpp
    test_fixtures.cpp
    test_roundtrip.cpp
    test_schema_files.cpp
    exact_eval.cpp
)
target_link_libraries(pcfb_tests PRIVATE pcfb Catch2::Catch2WithMain)
target_include_directories(pcfb_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/tools
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(pcfb_tests PRIVATE
    PCFB_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
    PCFB_SCHEMA_DIR="${PROJECT_SOURCE_DIR}/schema"
)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(pcfb_tests)
