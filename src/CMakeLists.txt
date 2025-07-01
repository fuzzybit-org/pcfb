FetchContent_Declare(
        flatbuffers
        GIT_REPOSITORY https://github.com/google/flatbuffers.git
        GIT_TAG v25.2.10 # Use a specific stable tag or branch
)
FetchContent_MakeAvailable(flatbuffers)

# Define the input and output files
set(FBS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../schema/pcfb_v1.fbs)
set(GENERATED_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/../include/pcfb_v1_generated.h)

# Add custom command to generate C++ files from FlatBuffer schema
add_custom_command(
        OUTPUT ${GENERATED_HEADER}
        COMMAND flatc --cpp --scoped-enums --gen-mutable --gen-object-api -o "${CMAKE_CURRENT_SOURCE_DIR}/../include/" ${FBS_FILE}
        DEPENDS ${FBS_FILE} flatc
        COMMENT "Generating PCFB C++ headers from FlatBuffer schema"
        VERBATIM
)

# Define the schema binary output file
set(SCHEMA_BINARY ${CMAKE_CURRENT_SOURCE_DIR}/../schema/pcfb_v1.bfbs)

# Add custom command to generate binary schema from FlatBuffer schema
add_custom_command(
        OUTPUT ${SCHEMA_BINARY}
        COMMAND flatc -b --schema -o "${CMAKE_CURRENT_SOURCE_DIR}/../schema/" ${FBS_FILE}
        DEPENDS ${FBS_FILE} flatc
        COMMENT "Generating PCFB binary schema from FlatBuffer schema"
        VERBATIM
)

# Create a target that depends on the generated files
add_custom_target(pcfb_cpp_headers
        DEPENDS ${GENERATED_HEADER}
        COMMENT "Generate PCFB C++ FlatBuffer headers"
)

# Create a target that generates the binary schema
add_custom_target(pcfb_schema
        DEPENDS ${SCHEMA_BINARY}
        COMMENT "Generate PCFB binary schema"
)

# -----------------------------------------------------------------------------
# Static library with implementation sources
# -----------------------------------------------------------------------------

set(PCFB_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/io.cpp
)

add_library(pcfb STATIC ${PCFB_SRC})

# Generated headers must be available before compilation
add_dependencies(pcfb pcfb_cpp_headers)

target_include_directories(pcfb
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(pcfb PUBLIC flatbuffers)