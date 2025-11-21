# Mid-360 Driver

This is an implementation of the Mid-360 driver, intended to serve as a replacement for [livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2).

<img src="./img/ACE.jpg" width="200px">

## Install dependencies

1. Please make sure you have install ROS2.
2. Install Asio. If you are using ubuntu, you can install by following command: `sudo apt install libasio-dev`

## Param

here are some parameters you can set in config file:

```yaml
mid360_driver:
    ros__parameters:
        lidar_topic: /livox/lidar
        lidar_frame: livox_frame
        imu_topic: /livox/imu
        imu_frame: imu_frame
        lidar_publish_time_interval: 0.1
        is_topic_name_with_lidar_ip: false # 是否在话题名后面加雷达ip，可以用于区分多个雷达
```

## Contact

QQ group: 1070252119

Email: 1709185482@qq.com

## License

Copyright (C) 2025 Yingjie Huang

Licensed under the MIT License. See License.txt in the project root for license information.
