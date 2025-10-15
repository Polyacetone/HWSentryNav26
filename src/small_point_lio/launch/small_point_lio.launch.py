from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    small_point_lio_node = Node(
        package="small_point_lio",
        executable="small_point_lio_node",
        name="small_point_lio",
        output="screen",
        parameters=[PathJoinSubstitution([
            FindPackageShare("small_point_lio"),
            "config",
            "mid360.yaml",
        ])],
    )

    return LaunchDescription([small_point_lio_node])
