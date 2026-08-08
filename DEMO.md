# HWSentryNav26 DEMO 说明

本文档介绍如何在不上车的情况下体验HWSentryNav26中部分模块的功能和效果。运行之前需要按照[README.md](./README.md)的说明进行环境配置。

本文章含人量100%，可放心食用。

## `small_glim` 里程计演示

1. 从[这里](https://drive.google.com/drive/folders/1FJgEmFS2wcuFFTUjiKikhG14Zo43saoj)下载`mid360_bag`录包。
2. 修改`small_glim`里程计的配置文件`params_node.yaml`：把`use_mapping_trigger`关掉，`enable_tf_publish`打开，设置`acc_scale`为`9.80665`。替换`imu_sub_topic`和`lidar_sub_topic`为录包中对应的IMU和Lidar话题`/livox/imu`和`/livox/lidar/pointcloud`。再修改`params_sensors.yaml`，把`T_lidar_imu`设置为`[0.011, 0.02329, -0.04412, 0.0, 0.0, 0.0, 1.0]`，`imu_acc_saturation_thresh`改成`39.0`。（因为实车上我使用的是下位机H7的IMU数据，录包是之前单独用MID360录制的，所以外参不同。）
3. 使用`ros2 launch small_glim small_glim.launch.py`启动里程计节点。使用`ros2 bag play mid360_bag`播放录包。然后打开Foxglove之类的可视化工具应该就能看到里程计话题了。
4. 建图结果默认保存在`~/mapping`（可以在`params_mapping.yaml`中修改）。`mapping.pcd`是全局点云，`frame_*.pcd`是关键帧点云，`poses.txt`是关键帧位置。

## `offline_mapping_optimizer` 离线建图优化演示

1. 使用`ros2 launch offline_mapping_optimizer offline_mapping_optimizer.launch.py data_path:=mapping_*`启动离线建图优化节点。其中`mapping_*`是上一步生成的建图结果文件夹。
2. 离线建图优化完成后，在刚刚的文件夹中输出`optimized_map.pcd`和`optimized_poses.txt`。不出意外的话，`optimized_map.pcd`应该比`mapping.pcd`更干净，中间的人影基本消失。

## `nav_executor` 导航演示

1. 从[这里](https://drive.google.com/drive/folders/1FJgEmFS2wcuFFTUjiKikhG14Zo43saoj)下载`RMUC202605.msgpack`和`RMUC202608af.pcd`全局地图文件，放到`map_server/maps`文件夹下，并重新构建。
2. 开非常多终端，分别启动：
   - `ros2 launch map_server map_server.launch.py`：提供全局地图/局部地图。
   - `ros2 launch nav_executor nav_executor.launch.py`：进行路径规划/路径跟随。
   - `ros2 launch tf_maintainer tf_maintainer.launch.py`：TF树维护节点。
   - `python3 HWSentryNav26/utils/py/sim/wheel_leg_lqr_follow_sim.py`：启动简单的路径跟随仿真节点。
   - `python3 HWSentryNav26/utils/py/msg/nav_goal_relay.py`：简单的话题转发节点，把`/nav_goal`话题（`geometry_msgs/PointStamped`类型）转发到`/nav_executor/nav_goal`（`interfaces/msg/NavGoal`类型），方便在Foxglove里点击导航目标。
3. 打开Foxglove之类的可视化工具，设置显示坐标系为`map`，查看`/map_server/debug/global_map_cloud`、`/nav_executor/debug/final_cost_map`、`/nav_executor/debug/minco_trajectory`、`/nav_executor/debug/mpc_path`话题和`odom`、`chassis_link`变换。然后在地图上点击目标发送`geometry_msgs/PointStamped`类型的`/nav_goal`话题，应该就能看到路径规划和路径跟随的效果了。
4. 如果想看动态避障效果，可以在`wheel_leg_lqr_follow_sim.py`里修改`OBSTACLE_SPECS`列表，添加一些障碍物。