cmake_minimum_required(VERSION 3.14)

project(LCT2)
set(CMAKE_CXX_STANDARD 14)

set(OpenCV_DIR "D:\\opencv\\OpenCV-MinGW-Build-OpenCV-4.1.1-x64")

find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS})

add_executable(LCT2 main.cpp LCT2tracker.h LCT2tracker.cpp assignToBins1.cpp assignToBins1.h detector.h util.h detector.cpp util.cpp fhog.cpp fhog.h)
target_link_libraries(LCT2 ${OpenCV_LIBS})
