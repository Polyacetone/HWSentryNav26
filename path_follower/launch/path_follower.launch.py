from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os, glob

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
    configs = glob.glob(os.path.join(get_package_share_directory('path_follower'), 'config', '*.yaml'))
    return LaunchDescription([
        Node(
            package="path_follower",
            executable="path_follower_node",
            output="screen",
            emulate_tty=True,
            parameters=configs,
            prefix=debug_prefix()
        )
    ])