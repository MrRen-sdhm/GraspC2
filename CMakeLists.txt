cmake_minimum_required(VERSION 3.8)
project(GraspC2_New)
#set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -lpthread -Wl,--no-as-needed  -g3 -Wall")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -fPIC -pthread -lpthread -Wl,--no-as-needed -g3 -Wall")
add_definitions( -fPIC )

# Opencv
#find_package(OpenCV 3.4 REQUIRED)
#include_directories(${OpenCV3_INCLUDE_DIRS})
#message(STATUS "OpenCV library status:")
#message(STATUS "    version: ${OpenCV_VERSION}")
#message(STATUS "    libraries: ${OpenCV3_LIBRARIES}")
#message(STATUS "    include path: ${OpenCV3_INCLUDE_DIRS}")

#set(CMAKE_VERBOSE_MAKEFILE ON)

#add 3rdparty lib
set(CBTC_ROOT "/opt/cobotsys")

#add cobotsys module
set(CMAKE_MODULE_PATH "/opt/cobotsys/cmake" ${CMAKE_MODULE_PATH})
find_package(cobotsys REQUIRED)

# PCL
find_package(PCL 1.8.1 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Libtorch
set(CMAKE_PREFIX_PATH "/home/hustac/libtorch1.0/share/cmake")
find_package(Torch)
message(STATUS "Pytorch status:")
message(STATUS "libraries: ${TORCH_LIBRARIES}")

# OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#add include directory
include_directories(${CobotCore_INCLUDE_DIRS})

#add  cmake out of source build
SET(EXAMPLES_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
STRING(TOLOWER ${CMAKE_BUILD_TYPE} COBOTSYS_EXAMPLES_BUILD_TYPE)
SET(COBOTEXAMPLES_OUTPUT_DIRECTORY_ROOT ${EXAMPLES_ROOT}/build/${COBOTSYS_EXAMPLES_BUILD_TYPE})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${COBOTEXAMPLES_OUTPUT_DIRECTORY_ROOT}/bin" CACHE PATH "Runtime directory" FORCE)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${COBOTEXAMPLES_OUTPUT_DIRECTORY_ROOT}/lib" CACHE PATH "Library directory" FORCE)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${COBOTEXAMPLES_OUTPUT_DIRECTORY_ROOT}/lib" CACHE PATH "Archive directory" FORCE)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals")

#add qt.conf
configure_file("${EXAMPLES_ROOT}/cmake/qt.conf.in"
        "${EXAMPLES_ROOT}/cmake/qt.conf" IMMEDIATE @ONLY)
file(COPY ${EXAMPLES_ROOT}/cmake/qt.conf DESTINATION ${EXAMPLES_ROOT}/build/${COBOTSYS_EXAMPLES_BUILD_TYPE}/bin)

#add subdirectory
add_subdirectory(app)
add_subdirectory(grasp)


