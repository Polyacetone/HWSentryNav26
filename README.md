# 2026轮腿哨兵导航

## 环境要求

> [!NOTE]
> 推荐使用Ubuntu24.04。

### ROS2-Jazzy
推荐使用[鱼香ROS自动安装脚本](https://github.com/fishros/install)。

### Eigen3
推荐使用apt安装：`sudo apt install libeigen3-dev`。

### OpenCV
推荐使用apt安装：`sudo apt install libopencv-dev`。

### PCL
推荐使用apt安装：`sudo apt install libpcl-dev ros-jazzy-pcl-ros`。

### Ceres
推荐使用apt安装：`sudo apt install libceres-dev`。

### Asio
推荐使用apt安装：`sudo apt install libasio-dev ros-jazzy-asio-cmake-module`。

### GTSAM
可能需要按照以下步骤自行构建。
```bash
sudo apt install libomp-dev libboost-all-dev libmetis-dev \
                 libfmt-dev libspdlog-dev \
                 libglm-dev libglfw3-dev libpng-dev libjpeg-dev
git clone https://github.com/borglab/gtsam
cd gtsam && git checkout 4.3a0
mkdir build && cd build
cmake .. -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
         -DGTSAM_BUILD_TESTS=OFF \
         -DGTSAM_WITH_TBB=OFF \
         -DGTSAM_USE_SYSTEM_EIGEN=ON \
         -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
         -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

### gtsam_points
可能需要按照以下步骤自行构建。
```bash
git clone https://github.com/koide3/gtsam_points
mkdir gtsam_points/build && cd gtsam_points/build
cmake .. -DBUILD_WITH_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

### CUDA（可选）
地图离线优化工具（map_optimizer）可以使用CUDA加速光束法去除动态障碍物。如果没有CUDA环境，可以禁用该功能，会回落到CPU实现。

CUDA可以使用apt安装：`sudo apt install nvidia-cuda-toolkit`。若需要更高版本的CUDA，可以参考NVIDIA官网的[安装指南](https://developer.nvidia.com/cuda-downloads)。