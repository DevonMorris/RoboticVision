cmake_minimum_required(VERSION 3.0)
project( DisplayWebcam )

set(CMAKE_CXX_STANDARD 11)


find_package( OpenCV REQUIRED )
add_executable(filter_webcam src/FilterWebcam.cpp )

target_link_libraries(filter_webcam ${OpenCV_LIBS})
