include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(perennis_tests
    ggml_smoke_test.cpp
)
target_link_libraries(perennis_tests PRIVATE perennis GTest::gtest_main)
target_compile_options(perennis_tests PRIVATE -Wno-conversion -Wno-sign-conversion)

include(GoogleTest)
gtest_discover_tests(perennis_tests)
