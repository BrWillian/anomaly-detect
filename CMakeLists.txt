cmake_minimum_required(VERSION 3.10)
project(AnomalyDetector)

find_package(OpenCV REQUIRED)

include_directories(/usr/include/eigen3)

add_executable(AnomalyDetector src/main.cpp
        src/AnomalyDetector.cpp
        src/wavelet.cpp
        src/matrix.cpp
        include/vector.h
)

target_link_libraries(AnomalyDetector ${OpenCV_LIBS})