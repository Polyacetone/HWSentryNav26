import rclpy
from rclpy.node import Node
from interfaces.msg import ChassisCmd
import time

class ChassisCmdPublisher(Node):
    def __init__(self):
        super().__init__('chassis_cmd_publisher')
        self.publisher_ = self.create_publisher(ChassisCmd, '/path_follower/chassis_cmd', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        msg = ChassisCmd()
        self.publisher_.publish(msg)
        self.get_logger().info('Published empty ChassisCmd message')

def main(args=None):
    rclpy.init(args=args)
    node = ChassisCmdPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()