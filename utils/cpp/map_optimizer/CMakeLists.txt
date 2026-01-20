cmake_minimum_required(VERSION 3.10)
project(map_optimizer)
set(EXECUTABLE_NAME ${PROJECT_NAME}_node)

option(MAP_OPTIMIZER_USE_CUDA "Enable CUDA-accelerated raycasting" ON)

set(CMAKE_C_COMPILER "gcc-13")
set(CMAKE_CXX_COMPILER "g++-13")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-O3)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

file(GLOB_RECURSE sources "src/*.cpp")
ament_auto_add_executable(${EXECUTABLE_NAME} ${sources})

target_include_directories(${EXECUTABLE_NAME} PRIVATE include)

find_package(OpenMP REQUIRED)
target_link_libraries(${EXECUTABLE_NAME} OpenMP::OpenMP_CXX)

find_package(Eigen3 REQUIRED)
target_link_libraries(${EXECUTABLE_NAME} Eigen3::Eigen)

find_package(PCL REQUIRED)
target_include_directories(${EXECUTABLE_NAME} PRIVATE ${PCL_INCLUDE_DIRS})
target_compile_definitions(${EXECUTABLE_NAME} PRIVATE ${PCL_DEFINITIONS})
target_link_libraries(${EXECUTABLE_NAME} ${PCL_LIBRARIES})

find_package(GTSAM 4.3 REQUIRED)
target_link_libraries(${EXECUTABLE_NAME} gtsam)

find_package(gtsam_points 1.2.0 REQUIRED)
target_link_libraries(${EXECUTABLE_NAME} gtsam_points::gtsam_points)

if(MAP_OPTIMIZER_USE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
  target_sources(${EXECUTABLE_NAME} PRIVATE src/raycasting_cuda.cu)
  target_compile_definitions(${EXECUTABLE_NAME} PRIVATE MAP_OPTIMIZER_USE_CUDA=1)
  set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CUDA_STANDARD 17)
  set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CUDA_STANDARD_REQUIRED ON)
  set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CUDA_EXTENSIONS OFF)
  target_link_libraries(${EXECUTABLE_NAME} CUDA::cudart)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CUDA_ARCHITECTURES native)
  endif()
endif()

ament_auto_package(
  USE_SCOPED_HEADER_INSTALL_DIR
  INSTALL_TO_SHARE
  config
  launch
)