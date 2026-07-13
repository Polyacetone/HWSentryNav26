cmake_minimum_required(VERSION 3.10)
project(mpc_tuner)

set(CMAKE_C_COMPILER "gcc-13")
set(CMAKE_CXX_COMPILER "g++-13")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(msgpack-cxx REQUIRED)

set(core_sources
  src/config_loader.cpp
  src/console_input.cpp
  src/episode_runner.cpp
  src/optimizer.cpp
  src/scenario_compiler.cpp
  src/scene_bundle.cpp
  src/scene_map.cpp
  src/scene_visualization.cpp
  src/wheel_leg_plant.cpp
)
ament_auto_add_library(mpc_tuner_core SHARED ${core_sources})
target_include_directories(mpc_tuner_core PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mpc_tuner_core Eigen3::Eigen Threads::Threads yaml-cpp msgpack-cxx ${OpenCV_LIBS})

ament_auto_add_executable(mpc_tuner_node src/mpc_tuner_node.cpp)
target_link_libraries(mpc_tuner_node mpc_tuner_core)

ament_auto_add_executable(mpc_scene_recorder_node src/scene_recorder_node.cpp)
target_link_libraries(mpc_scene_recorder_node mpc_tuner_core)

ament_auto_add_executable(mpc_scene_previewer_node src/scene_previewer_node.cpp)
target_link_libraries(mpc_scene_previewer_node mpc_tuner_core)

foreach(target mpc_tuner_core mpc_tuner_node mpc_scene_recorder_node mpc_scene_previewer_node)
  target_compile_options(${target} PRIVATE
    $<$<CONFIG:Debug>:-O0 -g -Wall -Wextra -Wpedantic -fsanitize=address,undefined>
    $<$<CONFIG:Release>:-O3 -DNDEBUG -Wall -Wextra -Wpedantic -flto=auto -march=native>
  )
  target_link_options(${target} PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    $<$<CONFIG:Release>:-flto=auto>
  )
endforeach()

ament_auto_package(
  USE_SCOPED_HEADER_INSTALL_DIR
  INSTALL_TO_SHARE
  config
  launch
  scenes
)
