
include_directories(include)

file(GLOB SOURCES
        src/GraphicsGrasp.cpp
        src/*
        )

add_executable(Grasp ${SOURCES} Grasp.cpp)
add_executable(Graphic src/GraphicsGrasp.cpp src/YoloDetector.cpp src/Darknet.cpp Graphic.cpp)
add_executable(Camera src/GraphicsGrasp.cpp src/YoloDetector.cpp src/Darknet.cpp Camera.cpp)

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

target_link_libraries(Camera
        ${CobotCore_LIBRARIES}
        ${PCL_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBRARIES}
        )


