cmake_minimum_required(VERSION 3.0)
project( DisplayWebcam )

set(CMAKE_CXX_STANDARD 11)


find_package( OpenCV REQUIRED )
add_executable(display_webcam src/DisplayWebcam.cpp )

target_link_libraries(display_webcam ${OpenCV_LIBS})
