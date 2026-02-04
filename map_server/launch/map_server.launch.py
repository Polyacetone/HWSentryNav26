import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def debug_prefix():
    asan_options = {
        "new_delete_type_mismatch": "0",
        "verify_asan_link_order": "0",
        "detect_odr_violation": "0",
        "suppressions": "asan.supp"
    }
    lsan_options = {
        "detect_leaks": "1",
        "suppressions": "lsan.supp"
    }
    asan_env = ":".join([f"{k}={v}" for k, v in asan_options.items()])
    lsan_env = ":".join([f"{k}={v}" for k, v in lsan_options.items()])
    return f"env ASAN_OPTIONS={asan_env} LSAN_OPTIONS={lsan_env}"

def generate_launch_description():
    config_yaml = os.path.join(get_package_share_directory('map_server'), 'config', 'params.yaml')
    return LaunchDescription([
        Node(
            package="map_server",
            executable="map_server_node",
            output="screen",
            emulate_tty=True,
            parameters=[config_yaml],
            prefix=debug_prefix()
        )
    ])