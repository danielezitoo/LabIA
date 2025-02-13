cmake_minimum_required(VERSION 3.10)
project(KeyPointsAlgorithmsOnCars)

find_package(OpenCV REQUIRED)

add_executable(progetto src/main.cpp)
target_link_libraries(progetto ${OpenCV_LIBS})

set_target_properties(progetto PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
