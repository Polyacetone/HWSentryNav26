cmake_minimum_required(VERSION 3.8)
project(common_libs)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_package(
  USE_SCOPED_HEADER_INSTALL_DIR
)