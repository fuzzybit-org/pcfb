set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "Do not build FlatBuffers tests" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "Do not install FlatBuffers" FORCE)
set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "Do not build flathash" FORCE)

FetchContent_Declare(
    flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v25.2.10
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(flatbuffers)

set(FBS_FILE ${PROJECT_SOURCE_DIR}/schema/pcfb.fbs)
set(FBS_V1_FILE ${PROJECT_SOURCE_DIR}/schema/history/pcfb_v1.fbs)
set(FBS_V1_1_FILE ${PROJECT_SOURCE_DIR}/schema/history/pcfb_v1_1.fbs)
set(PCFB_GEN_ROOT "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(GENERATED_HEADER "${PCFB_GEN_ROOT}/include/pcfb_generated.h")
set(SCHEMA_BINARY "${PCFB_GEN_ROOT}/schema/pcfb.bfbs")
set(VERSION_HEADER "${PCFB_GEN_ROOT}/include/pcfb/version.h")

file(MAKE_DIRECTORY "${PCFB_GEN_ROOT}/include/pcfb")
configure_file(
    ${PROJECT_SOURCE_DIR}/include/pcfb/version.h.in
    ${VERSION_HEADER}
    @ONLY
)

add_custom_command(
    OUTPUT ${GENERATED_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PCFB_GEN_ROOT}/include"
    COMMAND flatc --cpp --scoped-enums --gen-mutable --gen-object-api -o "${PCFB_GEN_ROOT}/include/" ${FBS_FILE}
    DEPENDS ${FBS_FILE} flatc
    COMMENT "Generating PCFB C++ headers from FlatBuffer schema"
    VERBATIM
)

add_custom_command(
    OUTPUT ${SCHEMA_BINARY}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PCFB_GEN_ROOT}/schema"
    COMMAND flatc -b --schema -o "${PCFB_GEN_ROOT}/schema/" ${FBS_FILE}
    DEPENDS ${FBS_FILE} flatc
    COMMENT "Generating PCFB binary schema from FlatBuffer schema"
    VERBATIM
)

add_custom_target(pcfb_cpp_headers
    DEPENDS ${GENERATED_HEADER} ${VERSION_HEADER}
    COMMENT "Generate PCFB C++ FlatBuffer headers"
)

add_custom_target(pcfb_schema
    DEPENDS ${SCHEMA_BINARY}
    COMMENT "Generate PCFB binary schema"
)

add_custom_target(pcfb_conform
    COMMAND flatc --conform ${FBS_V1_FILE} ${FBS_FILE}
    COMMAND flatc --conform ${FBS_V1_1_FILE} ${FBS_FILE}
    DEPENDS flatc ${FBS_FILE} ${FBS_V1_FILE} ${FBS_V1_1_FILE}
    COMMENT "Checking the canonical schema conforms to v1 and v1.1"
    VERBATIM
)

set(PCFB_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/validate.cpp
)

add_library(pcfb STATIC ${PCFB_SRC} ${GENERATED_HEADER})
set_source_files_properties(${GENERATED_HEADER} PROPERTIES GENERATED TRUE)
add_dependencies(pcfb pcfb_cpp_headers pcfb_conform)

target_include_directories(pcfb
    PUBLIC
        $<BUILD_INTERFACE:${PCFB_GEN_ROOT}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${flatbuffers_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# FetchContent's flatbuffers target is not exported. Keep the link
# BUILD_INTERFACE-only so `install(EXPORT pcfbTargets)` does not require it.
target_link_libraries(pcfb PRIVATE $<BUILD_INTERFACE:flatbuffers>)

set(PCFB_GENERATED_HEADER "${GENERATED_HEADER}" PARENT_SCOPE)
set(PCFB_GENERATED_SCHEMA "${SCHEMA_BINARY}" PARENT_SCOPE)
set(PCFB_VERSION_HEADER "${VERSION_HEADER}" PARENT_SCOPE)
