import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_tuner(context):
    arguments = [
        "--output",
        LaunchConfiguration("output_dir").perform(context),
        "--tuner-config",
        LaunchConfiguration("tuner_config").perform(context),
        "--nav-config-dir",
        LaunchConfiguration("nav_config_dir").perform(context),
    ]
    if LaunchConfiguration("smoke").perform(context).lower() in ("1", "true", "yes"):
        arguments.append("--smoke")
    return [
        Node(
            package="mpc_tuner",
            executable="mpc_tuner_node",
            output="screen",
            emulate_tty=True,
            arguments=arguments,
        )
    ]


def generate_launch_description():
    package_share = get_package_share_directory("mpc_tuner")
    nav_share = get_package_share_directory("nav_executor")
    return LaunchDescription(
        [
            DeclareLaunchArgument("output_dir", default_value="mpc_tuning_results"),
            DeclareLaunchArgument(
                "tuner_config",
                default_value=os.path.join(package_share, "config", "tuner.yaml"),
            ),
            DeclareLaunchArgument(
                "nav_config_dir", default_value=os.path.join(nav_share, "config")
            ),
            DeclareLaunchArgument("smoke", default_value="false"),
            OpaqueFunction(function=launch_tuner),
        ]
    )
