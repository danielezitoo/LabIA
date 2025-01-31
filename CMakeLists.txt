cmake_minimum_required(VERSION 3.10) # o una versione successiva
project(KeyPointsAlgorithmsOnCars)

find_package(OpenCV REQUIRED)

# Specifica il file sorgente e il nome dell'eseguibile
add_executable(progetto src/main.cpp)
target_link_libraries(progetto ${OpenCV_LIBS})

# Imposta la directory di output per l'eseguibile nella directory principale
set_target_properties(progetto PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)