find_package(OpenCV REQUIRED )
find_package(Eigen3 REQUIRED)

link_directories(${Eigen_INCLUDE_DIRS})

add_executable(VRStreaming VRStreaming.cc)

target_link_libraries (VRStreaming ${MULTISENSE_UTILITY_LIBS} ${OpenCV_LIBS} ${Eigen_LIBRARIES})

install(TARGETS VRStreaming RUNTIME DESTINATION "bin")
