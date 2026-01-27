# 2026轮腿哨兵导航

## 必要依赖

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

### Boost
推荐使用apt安装：`sudo apt install libboost-all-dev`。

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

## 可选依赖

utils里有一些小工具，不是导航功能必须的。如果需要使用这些工具，可能需要安装以下依赖。

### Python相关库
utils/py下有一些Python脚本，可能需要以下Python库。
推荐使用uv安装：`uv pip install numpy matplotlib scipy open3d scikit-learn pillow opencv-python`。

### absl（可选）
utils/cpp/offline_mapping_optimizer使用了absl库。如果不想安装absl也可以在CMake里禁用该功能。
推荐使用apt安装：`sudo apt install libabsl-dev`。

### CUDA（可选）
utils/cpp/offline_mapping_optimizer可以使用CUDA加速光束法去除动态障碍物。如果没有CUDA环境，可以禁用该功能，会回落到CPU实现。
CUDA可以使用apt安装：`sudo apt install nvidia-cuda-toolkit`。若需要更高版本的CUDA，可以参考NVIDIA官网的[安装指南](https://developer.nvidia.com/cuda-downloads)。