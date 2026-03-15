"""Relay a PointStamped goal into a NavGoal message.

Subscribes to a PointStamped goal on `/nav_goal` and republishes it as an
`interfaces/msg/NavGoal` on `/decision/nav_goal`. This is useful when upstream
nodes publish standard geometry messages but downstream nodes expect the
custom `NavGoal` message.

This node also allows configuring whether the relayed goal should be treated as
"fixed" (i.e. stay in place after reaching the goal).
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped
from interfaces.msg import NavGoal


class NavGoalRelay(Node):
    def __init__(self) -> None:
        super().__init__('nav_goal_relay')

        # ROS parameters (can be overridden via launch files or command line)
        self.declare_parameter('input_topic', '/nav_goal')
        self.declare_parameter('output_topic', '/decision/nav_goal')
        self.declare_parameter('fixed', False)

        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.fixed = bool(self.get_parameter('fixed').get_parameter_value().bool_value)

        self.pub = self.create_publisher(NavGoal, self.output_topic, 1)
        self.sub = self.create_subscription(
            PointStamped, self.input_topic, self._point_cb, 1
        )

        self.get_logger().info(
            f"nav_goal_relay running, subscribing to '{self.input_topic}' "
            f"and publishing NavGoal on '{self.output_topic}' (fixed={self.fixed})"
        )

    def _point_cb(self, msg: PointStamped) -> None:
        out = NavGoal()
        out.x = msg.point.x
        out.y = msg.point.y
        out.fixed = self.fixed
        self.pub.publish(out)
        self.get_logger().debug(
            f"relayed PointStamped ({out.x:.3f}, {out.y:.3f}) -> NavGoal(fixed={out.fixed})"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NavGoalRelay()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()