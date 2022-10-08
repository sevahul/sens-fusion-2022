cmake_minimum_required(VERSION 3.9)

project( stereo_estimation )

###############
# packages #
###############

FIND_PACKAGE(Boost COMPONENTS program_options filesystem REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(OpenMP)
find_package(jsoncpp REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})

###############
# executables #
###############

add_executable( OpenCV_stereo src/main.cpp )
target_link_libraries(OpenCV_stereo ${OpenCV_LIBS} )
TARGET_LINK_LIBRARIES(OpenCV_stereo ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(OpenCV_stereo ${JSONCPP_LIBRARIES})


if(OpenMP_CXX_FOUND)
    target_link_libraries(OpenCV_stereo OpenMP::OpenMP_CXX)
endif()
