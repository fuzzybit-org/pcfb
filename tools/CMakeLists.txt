add_executable(pcfb-make-fixtures make_fixtures.cpp)
target_link_libraries(pcfb-make-fixtures PRIVATE pcfb)
target_include_directories(pcfb-make-fixtures PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
