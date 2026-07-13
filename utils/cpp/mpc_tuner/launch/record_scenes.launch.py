import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    map_server_share = get_package_share_directory("map_server")
    nav_config_directory = os.path.join(
        get_package_share_directory("nav_executor"), "config"
    )

    recorder = Node(
        package="mpc_tuner",
        executable="mpc_scene_recorder_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            os.path.join(map_server_share, "config", "params.yaml"),
            {
                "split": LaunchConfiguration("split"),
                "output_directory": LaunchConfiguration("output_dir"),
                "map_directory": os.path.join(map_server_share, "maps"),
                "map_filename": LaunchConfiguration("map_filename"),
                "nav_config_directory": nav_config_directory,
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "split",
                description="Bundle split: train or validation",
            ),
            DeclareLaunchArgument(
                "output_dir",
                default_value=".",
                description="Directory where the timestamped bundle is written",
            ),
            DeclareLaunchArgument(
                "map_filename",
                description="Terrain map filename under map_server/maps",
            ),
            recorder,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=recorder,
                    on_exit=[Shutdown(reason="Scene recorder exited")],
                )
            ),
        ]
    )
