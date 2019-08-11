add_executable(CAssemblyC2Start CAssemblyC2Start.cpp ArmMoveTo.cpp)
target_link_libraries(CAssemblyC2Start  ${CobotCore_LIBRARIES} ${PCL_LIBRARIES} ${TORCH_LIBRARIES} ${OpenCV3_LIBRARIES})
#target_link_libraries(CAssemblyC2Start  ${CobotCore_LIBRARIES} ${PCL_LIBRARIES})

