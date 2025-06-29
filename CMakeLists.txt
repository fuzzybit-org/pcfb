FetchContent_Declare(
        flatbuffers
        GIT_REPOSITORY https://github.com/google/flatbuffers.git
        GIT_TAG v25.2.10 # Use a specific stable tag or branch
)
FetchContent_MakeAvailable(flatbuffers)

# Define the input and output files
set(FBS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/schema/pcfb_v1.fbs)
set(GENERATED_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/pcfb.h)

# Add custom command to generate C++ files from FlatBuffer schema
add_custom_command(
        OUTPUT ${GENERATED_HEADER}
        COMMAND flatc --cpp --scoped-enums --gen-mutable --gen-object-api -o ${CMAKE_CURRENT_SOURCE_DIR} ${FBS_FILE}
        DEPENDS ${FBS_FILE} flatc
        COMMENT "Generating PCFB C++ headers from FlatBuffer schema"
        VERBATIM
)

# Create a target that depends on the generated files
add_custom_target(pcfb_cpp_headers
        DEPENDS ${GENERATED_HEADER}
        COMMENT "Generate PCFB C++ FlatBuffer headers"
)

# Add the generated header to the pcfb library if you plan to create one
# Uncomment and modify as needed:
add_library(pcfb ${GENERATED_HEADER})
target_include_directories(pcfb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pcfb PUBLIC flatbuffers)