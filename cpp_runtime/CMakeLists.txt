cmake_minimum_required(VERSION 3.18)
project(guardian_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags matching Tesla's safety-critical requirements
add_compile_options(
    -Wall -Wextra -Wpedantic -Werror
    -fsanitize=address,thread    # AddressSanitizer + ThreadSanitizer
    -fno-omit-frame-pointer
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
)

add_link_options(
    -fsanitize=address,thread
)

# Main runtime executable
add_executable(guardian_runtime
    src/guardian_runtime.cpp
)
target_include_directories(guardian_runtime PRIVATE include)

# Unit tests
add_executable(test_spsc_ring_buffer
    tests/test_spsc_ring_buffer.cpp
)
target_include_directories(test_spsc_ring_buffer PRIVATE include)

# Benchmarks
add_executable(bench_runtime
    tests/bench_runtime.cpp
)
target_include_directories(bench_runtime PRIVATE include)
target_compile_options(bench_runtime PRIVATE -O3 -march=native)
