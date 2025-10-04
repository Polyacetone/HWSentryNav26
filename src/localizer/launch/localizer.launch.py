import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
def generate_launch_description():
    config_yaml = os.path.join(get_package_share_directory('localizer'), 'config', 'params.yaml')
    return LaunchDescription([
        Node(
            package="localizer",
            executable="localizer_node",
            output="screen",
            emulate_tty=True,
            parameters=[config_yaml],
        )
    ])