cmake_minimum_required (VERSION 3.24)

# Include helper scripts early so that BUILD_NUMBER / BUILD_COMMIT are available
include(${CMAKE_CURRENT_LIST_DIR}/cmake/build-info.cmake)

# Propagate a semantic version of the form 0.0.<commit-count>
project(pcfb VERSION 0.0.${BUILD_NUMBER} LANGUAGES C CXX)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/common-flags.cmake)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

include(CheckIncludeFileCXX)
option(BUILD_TESTS "Build tests" ON)  # Influences CTest.

Include(FetchContent)
# Build the library (implemented as a header-only INTERFACE target) in the src directory.
add_subdirectory(src)
if (BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif ()

# -----------------------------------------------------------------------------
# Install & package
# -----------------------------------------------------------------------------

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(PCFB_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header files")

# Public interface headers. The FlatBuffer generated header is created at build
# time, so we only list it for installation – it is generated before the
# install step thanks to the dependency chain.
set(PCFB_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pcfb.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pcfb_v1_generated.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pcfb/io.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pcfb/error.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pcfb/ir/circuit.h)

# Install the header-only INTERFACE target and its public headers.
install(TARGETS pcfb EXPORT pcfbTargets
        INCLUDES DESTINATION ${PCFB_INCLUDE_INSTALL_DIR})
install(FILES ${PCFB_PUBLIC_HEADERS} DESTINATION ${PCFB_INCLUDE_INSTALL_DIR})

# Export the target so that downstream projects can consume it via
# find_package(pcfb).
install(EXPORT pcfbTargets
        FILE pcfbTargets.cmake
        NAMESPACE pcfb::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pcfb)

# Generate and install the *Config*.cmake files
configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/pcfbConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/pcfbConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pcfb
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/pcfbConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/pcfbConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/pcfbConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pcfb)