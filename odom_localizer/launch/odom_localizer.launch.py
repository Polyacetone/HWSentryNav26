import os
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory

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
    config_yaml = os.path.join(get_package_share_directory('odom_localizer'), 'config', 'params.yaml')
    return LaunchDescription([
        ComposableNodeContainer(
            name='odom_localizer_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[
                ComposableNode(
                    package='odom_localizer',
                    plugin='odom_localizer::OdomLocalizerNode',
                    name='odom_localizer',
                    parameters=[config_yaml],
                )
            ],
            output='screen',
            emulate_tty=True,
            prefix=debug_prefix()
        )
    ])
