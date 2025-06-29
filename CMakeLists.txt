cmake_minimum_required (VERSION 3.24)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

project("pcfb" LANGUAGES C CXX)
include(CheckIncludeFileCXX)
option(BUILD_TESTS "Build tests" ON)  # Influences CTest.

Include(FetchContent)
add_subdirectory(src)
if (BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif ()

#
# install
#
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(PCFB_BUILD_NUMBER        ${BUILD_NUMBER})
set(PCFB_BUILD_COMMIT        ${BUILD_COMMIT})
set(PCFB_INSTALL_VERSION 0.0.${BUILD_NUMBER})

set(PCFB_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(PCFB_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(PCFB_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

set(PCFB_PUBLIC_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/pcfb.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/pcfb_flatbuffer.h)

set_target_properties(pcfb
        PROPERTIES
        PUBLIC_HEADER "${PCFB_PUBLIC_HEADERS}")

install(TARGETS pcfb LIBRARY PUBLIC_HEADER)