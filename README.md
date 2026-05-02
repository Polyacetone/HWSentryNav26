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
推荐按照以下步骤自行构建。
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
推荐按照以下步骤自行构建。
```bash
git clone https://github.com/koide3/gtsam_points
mkdir gtsam_points/build && cd gtsam_points/build
cmake .. -DBUILD_WITH_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MARCH_NATIVE=OFF
make -j$(nproc)
sudo make install
```

## 可选依赖

`utils`里有一些小工具，不是导航功能必须的。如果需要使用这些工具，可能需要安装以下依赖。

### Python相关库
`utils/py`下有一些Python脚本，可能需要以下Python库。
推荐使用uv安装：`uv pip install numpy matplotlib scipy open3d scikit-learn pillow opencv-python`。

### absl（可选）
`utils/cpp/offline_mapping_optimizer`使用了absl库。如果不想安装absl也可以在CMake里禁用该功能。
推荐使用apt安装：`sudo apt install libabsl-dev`。

### CUDA（可选）
`utils/cpp/offline_mapping_optimizer`可以使用CUDA加速光束法去除动态障碍物。如果没有CUDA环境，可以禁用该功能，会回落到CPU实现。
CUDA可以使用apt安装：`sudo apt install nvidia-cuda-toolkit`。若需要更高版本的CUDA，可以参考NVIDIA官网的[安装指南](https://developer.nvidia.com/cuda-downloads)。

## 开发提示

- 项目在Debug和Release模式下使用不同的编译选项。Debug模式下启用了`-fsanitize=address,undefined`以帮助检测内存错误和未定义行为，但是非常影响性能。正常开发和测试时建议使用Release模式，遇到问题时再切换到Debug模式进行排查。构建模式可以在`compile.sh`中修改。

- 在Debug模式下使用LSAN时，可能会误报`rcl_node_init`函数的内存泄漏。这是由于ROS2的节点初始化过程中分配了一些全局资源，LSAN无法正确识别这些资源的生命周期。建议将`leak:rcl_node_init`加入LSAN的忽略文件（默认存放在工作空间下的`lsan.supp`），以避免误报。

## 部署提示

- Ubuntu系统下建议关闭NTP服务以避免系统时间跳变导致的里程计和TF异常：`sudo timedatectl set-ntp false`。

- 如果运行时遇到报错`what():  Could not load library dlopen error: /lib/x86_64-linux-gnu/libpcl_io.so.1.14: undefined symbol: libusb_set_option, at ./src/shared_library.c:99`，这是海康驱动干的。直接删除`/opt/MVS/lib/64/libusb-1.0.so.0`和`/opt/MVS/lib/32/libusb-1.0.so.0`即可。

- 可以关闭网卡节能模式以避免网络不稳定。`sudo nano /etc/NetworkManager/conf.d/default-wifi-powersave-on.conf`，将`wifi.powersave`的值改为2即可。