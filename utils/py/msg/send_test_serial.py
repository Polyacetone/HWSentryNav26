import rclpy
from rclpy.node import Node
from interfaces.msg import ChassisCmd
from interfaces.msg import ShootCmd
import time

class ChassisCmdPublisher(Node):
    def __init__(self):
        super().__init__('chassis_cmd_publisher')
        self.chassis_cmd_publisher_ = self.create_publisher(ChassisCmd, '/path_follower/chassis_cmd', 10)
        self.shoot_cmd_publisher_ = self.create_publisher(ShootCmd, '/autoaim_controller/shoot_cmd', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        msg = ChassisCmd()
        msg.velocity = 114.0
        msg.theta = 1.14
        msg.omega = 514.0
        msg.step_up_ahead = True
        msg.step_down_ahead = False
        msg.slow_spin = True
        msg.fast_spin = False
        self.chassis_cmd_publisher_.publish(msg)
        self.get_logger().info('Published ChassisCmd message')
        msg = ShootCmd()
        msg.yaw = 0.514
        msg.pitch = 1.14
        msg.shoot_interval = -1
        self.shoot_cmd_publisher_.publish(msg)
        self.get_logger().info('Published ShootCmd message')

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