from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os, glob
def generate_launch_description():
    configs = glob.glob(os.path.join(get_package_share_directory('small_glim'), 'config', 'params_*.yaml'))
    return LaunchDescription([
        Node(
            package="small_glim",
            executable="small_glim_node",
            output="screen",
            emulate_tty=True,
            parameters=configs,
        )
    ])