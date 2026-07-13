import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scene_directory = os.path.join(
        get_package_share_directory("mpc_tuner"), "scenes"
    )
    previewer = Node(
        package="mpc_tuner",
        executable="mpc_scene_previewer_node",
        output="screen",
        emulate_tty=True,
        parameters=[{"scene_directory": LaunchConfiguration("scene_dir")}],
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "scene_dir",
                default_value=scene_directory,
                description="Directory recursively containing scene bundle files",
            ),
            previewer,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=previewer,
                    on_exit=[Shutdown(reason="Scene previewer exited")],
                )
            ),
        ]
    )
