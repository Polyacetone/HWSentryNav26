from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os, glob
def generate_launch_description():
    configs = glob.glob(os.path.join(get_package_share_directory('mid360_driver'), 'config', 'params.yaml'))
    return LaunchDescription([
        Node(
            package="mid360_driver",
            executable="mid360_driver_node",
            output="screen",
            emulate_tty=True,
            parameters=configs,
        )
    ])