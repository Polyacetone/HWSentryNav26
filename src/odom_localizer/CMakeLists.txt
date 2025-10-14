cmake_minimum_required(VERSION 3.10)
project(odom_localizer)

set(CMAKE_C_COMPILER "gcc-13")
set(CMAKE_CXX_COMPILER "g++-13")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-O3 -march=native -flto=auto)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

file(GLOB sources "src/*.cpp")
ament_auto_add_library(${PROJECT_NAME} SHARED ${sources})

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN odom_localizer::OdomLocalizerNode
  EXECUTABLE ${PROJECT_NAME}_node
)

ament_auto_package(
  USE_SCOPED_HEADER_INSTALL_DIR
  INSTALL_TO_SHARE
  config
  launch
)