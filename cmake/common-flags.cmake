# -march=native binds pcfb (and everything configured below this directory,
# e.g. FetchContent flatbuffers) to the build host's ISA; binaries built on a
# newer CPU then SIGILL on older ones. Keep it opt-in for local benchmarking.
option(PCFB_OPTIMIZE_FOR_NATIVE "Compile pcfb with -march=native (non-portable)" OFF)

if(PCFB_OPTIMIZE_FOR_NATIVE)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()
