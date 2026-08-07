<div align="center">

# HWSentryNav26

**面向 RoboMaster 轮腿哨兵机器人的 ROS2 导航框架**

</div>

# 项目简介

本项目是浙江大学 Hello World 战队在 RoboMaster 2026 的轮腿哨兵机器人上使用的导航框架。该框架针对轮腿哨兵机器人的特殊需求进行了定制化设计，能够有效处理复杂地形下的导航问题。

主要亮点包括：

1. **基于因子图优化的鲁棒里程计**。本项目在开源 SLAM 框架 [GLIM](https://github.com/koide3/glim) 的基础上进行了定制化开发。针对平衡轮腿哨兵剧烈撞击、翻车等极端工况，采用因子图滑窗优化能在平滑窗口内联合优化历史位姿，显著提升里程计的鲁棒性。
2. **针对跨地形场景的路径规划策略**。针对平衡轮腿机器人跨地形时的速度和方向约束，按“从粗到细”四段推进。首先用空间 A\* 决定地形通道。基于空间路径提取走廊，用曲率弧长运动基元做 Kinodynamic A\* 动力学搜索。然后对粗动力学可行路径进行分段五次多项式 MINCO 连续优化。最后在弧长域求解速度剖面。
3. **LPV 隐状态模型 + FDDP 模型预测控制**。轮腿底盘纵向响应是非最小相位系统，普通 PID 或者简单模型 MPC 控制效果差。本项目基于实车录制的命令-响应数据离线辨识以底盘高度为调度变量的 LPV 模型，不可直接测量的隐状态由观测器在线估计，配合管式辅助环抑制模型误差。MPC 求解器为自研 FDDP，性能较好。
4. **自主设计的导航状态机**。导航状态机采用扁平化有限状态结构，便于管理和维护。状态机为本项目从零设计，未采用 Navigation2 行为树等现成方案，灵活度较高。
5. **聚类动态障碍物跟踪与多步预测代价图**。点云差分提取动态点后，经形态学闭运算与连通域聚类为独立目标，匈牙利最优匹配完成帧间关联，卡尔曼恒速滤波维护航迹，并以局部足迹栅格保留目标形状。确认航迹按衰减速度模型外推，生成逐帧代价图序列。预测帧按时间对齐进入 MPC 预测代价，实现动态障碍物的预测避让。
6. **因子图离线建图优化与动态物体去除**。在线里程计受实时性限制仅做固定滞后平滑，全局一致性可能不足。离线优化器将关键帧组织为两阶段因子图。局部子图内帧间 VGICP 配准优化，全局以子图为节点再做 VGICP + 回环检测，获得全局一致的位姿轨迹。随后用 3D-DDA 光线投射自动去除动态物体，产出干净的全局点云供定位与动态检测使用。

更详细的设计说明请参见 [架构设计概述](./DESIGN.md)。

想在不上车的情况下品鉴效果，请参考 [DEMO 说明](./DEMO.md)。

后续可能继续更新完善文档，点个 Star ⭐ 谢谢喵。

# 环境配置

## 必要依赖

### OS

推荐使用Ubuntu 24.04 LTS，安装ROS2-Jazzy比较方便。

### ROS2-Jazzy

推荐使用[鱼香ROS自动安装脚本](https://github.com/fishros/install)。

### Eigen3

推荐使用apt安装：`sudo apt install libeigen3-dev`。

### OpenCV

推荐使用apt安装：`sudo apt install libopencv-dev ros-jazzy-cv-bridge`。

### PCL

推荐使用apt安装：`sudo apt install libpcl-dev ros-jazzy-pcl-ros`。

### Asio

推荐使用apt安装：`sudo apt install libasio-dev ros-jazzy-asio-cmake-module`。

### Boost

推荐使用apt安装：`sudo apt install libboost-all-dev`。

### MsgPack

推荐使用apt安装：`sudo apt install libmsgpack-cxx-dev`。

### GTSAM

推荐按照以下步骤自行构建。

```bash
sudo apt install libomp-dev libboost-all-dev libmetis-dev libfmt-dev libspdlog-dev
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
cd gtsam_points && git checkout v1.2.0
mkdir build && cd build
cmake .. -DBUILD_WITH_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MARCH_NATIVE=OFF
make -j$(nproc)
sudo make install
```

### [HWSentryCommon26](https://github.com/Polyacetone/HWSentryCommon26)

本项目的构建依赖其中自定义消息`interfaces`包和共用工具库`common_libs`。

推荐将HWSentryCommon26和HWSentryNav26放在同一工作空间下构建，例如：

```bash
mkdir ~/hwsentry_ws && cd ~/hwsentry_ws
git clone https://github.com/Polyacetone/HWSentryCommon26
git clone https://github.com/Polyacetone/HWSentryNav26
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

## 可选依赖

`utils/py`下有一些Python脚本，可能需要以下Python库。推荐使用uv安装：`uv pip install numpy matplotlib scipy open3d pillow opencv-python msgpack pyyaml numba mcap mcap-ros2-support cadquery`。

# 提示

## 开发提示

- 项目在Debug和Release模式下使用不同的编译选项。Debug模式下启用了`-fsanitize=address,undefined`以帮助检测内存错误和未定义行为，但是非常影响性能。正常开发和测试时建议使用Release模式，遇到问题时再切换到Debug模式进行排查。构建模式可以在`compile.sh`中修改。

- 在Debug模式下使用LSAN时，可能会误报`rcl_node_init`函数的内存泄漏。这是由于ROS2的节点初始化过程中分配了一些全局资源，LSAN无法正确识别这些资源的生命周期。建议将`leak:rcl_node_init`加入LSAN的忽略文件（默认存放在工作空间下的`lsan.supp`），以避免误报。

## 部署提示

- Ubuntu系统下建议关闭NTP服务以避免系统时间跳变导致的里程计和TF异常：`sudo timedatectl set-ntp false`。

- 如果运行时遇到报错`Could not load library dlopen error: /lib/x86_64-linux-gnu/libpcl_io.so.1.14: undefined symbol: libusb_set_option, at ./src/shared_library.c:99`，这是海康驱动干的。直接删除`/opt/MVS/lib/64/libusb-1.0.so.0`和`/opt/MVS/lib/32/libusb-1.0.so.0`即可。

- 可以关闭网卡节能模式以避免网络不稳定导致远程调车卡顿。`sudo nano /etc/NetworkManager/conf.d/default-wifi-powersave-on.conf`，将`wifi.powersave`的值改为2即可。