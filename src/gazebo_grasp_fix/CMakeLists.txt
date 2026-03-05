cmake_minimum_required(VERSION 3.8)
project(gazebo_grasp_fix)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(std_srvs REQUIRED)

add_library(gazebo_grasp_fix SHARED src/gazebo_grasp_fix.cpp)

target_include_directories(gazebo_grasp_fix PRIVATE ${GAZEBO_INCLUDE_DIRS})
target_link_libraries(gazebo_grasp_fix ${GAZEBO_LIBRARIES})

ament_target_dependencies(gazebo_grasp_fix rclcpp gazebo_dev gazebo_ros std_srvs)

install(TARGETS gazebo_grasp_fix
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_package()
