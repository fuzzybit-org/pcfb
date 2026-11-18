cmake_minimum_required (VERSION 3.24)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/build-info.cmake)

project(pcfb VERSION 0.0.${BUILD_NUMBER} LANGUAGES C CXX)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/common-flags.cmake)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Legacy aliases for the CTest-standard BUILD_TESTING option.
if (DEFINED PCFB_BUILD_TESTS AND NOT DEFINED BUILD_TESTING)
    set(BUILD_TESTING ${PCFB_BUILD_TESTS})
endif ()
if (DEFINED BUILD_TESTS AND NOT DEFINED BUILD_TESTING)
    set(BUILD_TESTING ${BUILD_TESTS})
endif ()

include(CTest)

include(FetchContent)

add_subdirectory(src)

if (BUILD_TESTING)
    add_subdirectory(tests)
    add_subdirectory(tools)
endif ()

# -----------------------------------------------------------------------------
# Install & package
# -----------------------------------------------------------------------------

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(PCFB_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header files")

install(TARGETS pcfb EXPORT pcfbTargets
        INCLUDES DESTINATION ${PCFB_INCLUDE_INSTALL_DIR})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/pcfb.h
              ${PCFB_GENERATED_HEADER}
        DESTINATION ${PCFB_INCLUDE_INSTALL_DIR})
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pcfb/io.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pcfb/error.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pcfb/builder.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pcfb/validate.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pcfb/threshold.h
    ${PCFB_VERSION_HEADER}
    DESTINATION ${PCFB_INCLUDE_INSTALL_DIR}/pcfb)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/pcfb/ir/circuit.h
        DESTINATION ${PCFB_INCLUDE_INSTALL_DIR}/pcfb/ir)

install(EXPORT pcfbTargets
        FILE pcfbTargets.cmake
        NAMESPACE pcfb::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pcfb)

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
