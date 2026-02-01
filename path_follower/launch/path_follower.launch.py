from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os, glob
def generate_launch_description():
    configs = glob.glob(os.path.join(get_package_share_directory('path_follower'), 'config', '*.yaml'))
    return LaunchDescription([
        Node(
            package="path_follower",
            executable="path_follower_node",
            output="screen",
            emulate_tty=True,
            parameters=configs,
        )
    ])