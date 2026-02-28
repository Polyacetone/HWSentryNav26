from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os, glob
def generate_launch_description():
    config_yaml = os.path.join(get_package_share_directory('offline_mapping_optimizer'), 'config', 'params.yaml')
    data_path = LaunchConfiguration('data_path')
    return LaunchDescription([
        DeclareLaunchArgument(
            'data_path',
            description='Data path of raw mapping frames'
        ),
        Node(
            package="offline_mapping_optimizer",
            executable="offline_mapping_optimizer_node",
            output="screen",
            emulate_tty=True,
            parameters=[config_yaml, {'data_path': data_path}],
        )
    ])