
include_directories(include)

file(GLOB SOURCES
        src/GraphicsGrasp.cpp
        src/*
        )

add_executable(Grasp ${SOURCES} Grasp.cpp)
add_executable(Graphic src/GraphicsGrasp.cpp src/YoloLibtorchDetector.cpp src/Darknet.cpp Graphic.cpp)

target_link_libraries(Grasp
        ${CobotCore_LIBRARIES}
        ${PCL_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBRARIES}
        )

target_link_libraries(Graphic
        ${CobotCore_LIBRARIES}
        ${PCL_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBRARIES}
        )


