import math
import time

import rclpy
from rclpy.node import Node, QoSProfile

import tf2_ros
from geometry_msgs.msg import PointStamped
from interfaces.msg import ChassisCmd


class PatrolNavGoal(Node):
    """Publishes a PointStamped goal for simple A/B patrol.

    The node alternates between "point A" and "point B" whenever it
    determines that the robot has reached the current goal.  The criteria
    for switching are:

    * the current position (looked up via TF from `map` to
      `chassis_link`) is within a configurable tolerance of the goal point,
      **and**
    * the last chassis command message has both linear and angular
      velocity equal to zero.

    All of the important values (topic names, goal coordinates, tolerance,
    etc.) are declared as ROS2 parameters so they can be overridden using
    `ros2 run ... __params:=...` or a launch file.
    """

    def __init__(self) -> None:
        super().__init__('patrol_nav_goal')

        # parameters with defaults
        self.declare_parameter('topic', '/decision/nav_goal')
        # frames used for TF lookup instead of odometry
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'chassis_link')
        self.declare_parameter('chassis_cmd_topic', '/nav_executor/chassis_cmd')

        self.declare_parameter('point_a_x', 3.0)
        self.declare_parameter('point_a_y', 6.0)
        self.declare_parameter('point_b_x', 9.0)
        self.declare_parameter('point_b_y', 6.0)
        self.declare_parameter('tolerance', 0.5)  # metres

        topic = self.get_parameter('topic').get_parameter_value().string_value
        map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        cmd_topic = self.get_parameter('chassis_cmd_topic').get_parameter_value().string_value

        self.point_a = (
            float(self.get_parameter('point_a_x').value),
            float(self.get_parameter('point_a_y').value),
        )
        self.point_b = (
            float(self.get_parameter('point_b_x').value),
            float(self.get_parameter('point_b_y').value),
        )
        self.tolerance = float(self.get_parameter('tolerance').value)

        # store frame names for later
        self.map_frame = map_frame
        self.base_frame = base_frame

        self.goal_pub = self.create_publisher(PointStamped, topic, 1)
        self.cmd_sub = self.create_subscription(
            ChassisCmd, cmd_topic, self._cmd_cb, 1
        )

        # prepare TF buffer/listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # will query TF each timer tick instead of storing pose
        self._last_cmd: ChassisCmd | None = None
        self._current_goal_name = 'A'  # start with A

        # check periodically whether it's time to switch goal
        self.create_timer(0.1, self._timer_cb)

        self.get_logger().info(
            f"patrol_nav_goal running, publishing to '{topic}', "
            f"base_frame='{base_frame}' in map='{map_frame}', "
            f"chassis_cmd='{cmd_topic}'"
        )

        time.sleep(0.5)

        # immediately send A so the patrol starts
        self.send_goal(self.point_a)

    def _cmd_cb(self, msg: ChassisCmd) -> None:
        self._last_cmd = msg

    def _timer_cb(self) -> None:
        # need recent chassis command
        if self._last_cmd is None:
            return

        # try to lookup the transform; if unavailable just skip this tick
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ExtrapolationException,
            tf2_ros.ConnectivityException,
        ):
            return

        curr_x = tf.transform.translation.x
        curr_y = tf.transform.translation.y

        # compute distance to the active goal
        target = self.point_a if self._current_goal_name == 'A' else self.point_b
        dx = curr_x - target[0]
        dy = curr_y - target[1]
        dist = math.hypot(dx, dy)

        if dist <= self.tolerance and \
           self._last_cmd.velocity == 0.0 and \
           self._last_cmd.omega == 0.0:
            # reached and stopped, flip to the other point
            self._current_goal_name = 'B' if self._current_goal_name == 'A' else 'A'
            new_target = self.point_a if self._current_goal_name == 'A' else self.point_b
            self.send_goal(new_target)

    def send_goal(self, point: tuple[float, float]) -> None:
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.point.x = point[0]
        msg.point.y = point[1]
        msg.point.z = 0.0
        self.goal_pub.publish(msg)
        self.get_logger().info(f"published nav_goal {point} (now goal {self._current_goal_name})")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PatrolNavGoal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
