cmake_minimum_required(VERSION 3.10)
project(path_planner)

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

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${CERES_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${CERES_LIBRARIES})

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN path_planner::PathPlannerNode
  EXECUTABLE ${PROJECT_NAME}_node
)

ament_auto_package(
  USE_SCOPED_HEADER_INSTALL_DIR
  INSTALL_TO_SHARE
  config
  launch
)