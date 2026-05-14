import rclpy
from rclpy.node import Node
from interfaces.msg import DecisionGimbalCmd
import math

def deg_to_rad(deg):
    return deg * math.pi / 180.0

class DecisionGimbalCmdPublisher(Node):
    def __init__(self):
        super().__init__('decision_gimbal_cmd_publisher')
        self.decision_gimbal_cmd_publisher_ = self.create_publisher(DecisionGimbalCmd, '/decision/decision_gimbal_cmd', 1)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.counter = 0

    def timer_callback(self):
        self.counter += 1
        msg = DecisionGimbalCmd()
        msg.pitch = deg_to_rad(-30.0)
        msg.yaw = deg_to_rad((self.counter * 10) % 360)
        self.decision_gimbal_cmd_publisher_.publish(msg)
        self.get_logger().info('Published DecisionGimbalCmd message')

def main(args=None):
    rclpy.init(args=args)
    node = DecisionGimbalCmdPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()