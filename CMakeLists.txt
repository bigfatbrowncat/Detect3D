# the minimum version of CMake required
cmake_minimum_required(VERSION 2.8)

project (Detect3D)

add_subdirectory(thirdparty/opencv)

include_directories(${OpenCV_BINARY_DIR}/include)
include_directories(${OpenCV_SOURCE_DIR}/include)
include_directories(${OpenCV_SOURCE_DIR}/modules/core/include)
include_directories(${OpenCV_SOURCE_DIR}/modules/imgcodecs/include)
include_directories(${OpenCV_SOURCE_DIR}/modules/imgproc/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(detect3d
	main.cpp
)

target_link_libraries (detect3d opencv_core opencv_imgcodecs opencv_imgproc)
