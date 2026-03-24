import rclpy
from rclpy.node import Node
from interfaces.msg import ChassisCmd
from interfaces.msg import ShootCmd
import math

class ChassisCmdPublisher(Node):
    def __init__(self):
        super().__init__('chassis_cmd_publisher')
        self.chassis_cmd_publisher_ = self.create_publisher(ChassisCmd, '/path_follower/chassis_cmd', 1)
        self.shoot_cmd_publisher_ = self.create_publisher(ShootCmd, '/autoaim_controller/shoot_cmd', 1)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        msg = ChassisCmd()

        msg.velocity = 0.0
        msg.omega = 0.0
        msg.step_up_ahead = False
        msg.step_down_ahead = False
        msg.slow_spin = False
        msg.fast_spin = False
        self.chassis_cmd_publisher_.publish(msg)
        self.get_logger().info('Published ChassisCmd message')

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