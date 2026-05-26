cmake_minimum_required(VERSION 3.20)
project(f1_embedded CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Homebrew mosquitto (Apple Silicon path)
set(MOSQUITTO_PREFIX "/opt/homebrew/opt/mosquitto")
set(CJSON_PREFIX "/opt/homebrew/opt/cjson")
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${MOSQUITTO_PREFIX}/include
    ${CJSON_PREFIX}/include
)
link_directories(${MOSQUITTO_PREFIX}/lib)

# Common sources (exclude mains)
set(COMMON_SRCS
    src/can_simulator.cpp
    src/mqtt_publisher.cpp
)

# Main telemetry daemon
add_executable(rtos_scheduler
    src/rtos_scheduler.cpp
    ${COMMON_SRCS}
)
target_link_libraries(rtos_scheduler mosquitto pthread)
target_compile_options(rtos_scheduler PRIVATE -O2 -Wall -Wextra -Wpedantic)

# WCET benchmark
add_executable(benchmark
    src/benchmark.cpp
    ${COMMON_SRCS}
)
target_link_libraries(benchmark mosquitto pthread)
target_compile_options(benchmark PRIVATE -O2 -Wall -Wextra -Wpedantic)
