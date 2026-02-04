cmake_minimum_required(VERSION 3.10)
project(path_follower)

set(CMAKE_C_COMPILER "gcc-13")
set(CMAKE_CXX_COMPILER "g++-13")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

file(GLOB sources "src/*.cpp")
ament_auto_add_library(${PROJECT_NAME} SHARED ${sources})

find_package(Boost REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${Boost_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)

find_package(Eigen3 REQUIRED)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

find_package(OpenCV REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

find_package(Ceres REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${CERES_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${CERES_LIBRARIES})

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN path_follower::PathFollowerNode
  EXECUTABLE ${PROJECT_NAME}_node
)

target_compile_options(${PROJECT_NAME} PRIVATE
  $<$<CONFIG:Debug>:-O0 -g -Wall -Wextra -Wpedantic -Wconversion -fsanitize=address,undefined -fstack-protector-strong -fcf-protection=full>
  $<$<CONFIG:Release>:-O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -flto=auto -march=native>
)

target_link_options(${PROJECT_NAME} PRIVATE
  $<$<CONFIG:Debug>:-fsanitize=address,undefined>
  $<$<CONFIG:Release>:-flto=auto>
)

target_compile_options(${PROJECT_NAME}_node PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
target_link_options(${PROJECT_NAME}_node PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)

ament_auto_package(
  USE_SCOPED_HEADER_INSTALL_DIR
  INSTALL_TO_SHARE
  config
  launch
)