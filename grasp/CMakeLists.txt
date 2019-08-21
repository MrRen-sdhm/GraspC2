
include_directories(include)

# GPD
find_library(GPD_POINTNET_LIBRARY NAMES gpd_pointnet PATHS /usr/local/lib PATH_SUFFIXES lib NO_DEFAULT_PATH)
if (GPD_POINTNET_LIBRARY)
    message(STATUS "Library GPD_POINTNET found in ${GPD_POINTNET_LIBRARY}")
else()
    message(FATAL_ERROR "Library GPD_POINTNET not found")
endif()
include_directories(${GPD_POINTNET_LIBRARY_INCLUDE_DIR})
message(STATUS "gpd_pointnet_include_dir: ${GPD_POINTNET_LIBRARY_INCLUDE_DIR}")

file(GLOB SOURCES
        src/GraphicsGrasp.cpp
        src/*
        )

add_executable(GraspC2 ${SOURCES} GraspC2.cpp)
add_executable(Graphic src/GraphicsGrasp.cpp src/YoloDetector.cpp src/Darknet.cpp Graphic.cpp)
add_executable(Camera src/GraphicsGrasp.cpp src/YoloDetector.cpp src/Darknet.cpp Camera.cpp)

target_link_libraries(GraspC2
        ${CobotCore_LIBRARIES}
        ${PCL_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBRARIES}
        #        ${GPD_POINTNET_LIBRARY}
        )

target_link_libraries(Graphic
        ${CobotCore_LIBRARIES}
        ${PCL_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBRARIES}
        )

target_link_libraries(Camera
        ${CobotCore_LIBRARIES}
        ${PCL_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBRARIES}
        )


