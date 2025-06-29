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

# Create a target that depends on the generated files
add_custom_target(pcfb_cpp_headers
        DEPENDS ${GENERATED_HEADER}
        COMMENT "Generate PCFB C++ FlatBuffer headers"
)

# -----------------------------------------------------------------------------
# Header-only INTERFACE library exposing the generated FlatBuffer header
# -----------------------------------------------------------------------------

# Build interface target and attach the header-generation step to it.
add_library(pcfb INTERFACE)

# Ensure the FlatBuffer code is generated before anything depends on the
# interface. This is enough because INTERFACE libraries don't participate in
# the build graph directly, so we make the dependency explicit here.
add_dependencies(pcfb pcfb_cpp_headers)

target_include_directories(pcfb INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(pcfb INTERFACE flatbuffers)