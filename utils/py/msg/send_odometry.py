import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import Odometry

class FakeOdomPublisher(Node):
    def __init__(self):
        super().__init__('fake_odom_publisher')
        self.odom_publisher_ = self.create_publisher(Odometry, '/small_glim/odometry', 1)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('FakeOdomPublisher started.')

    def timer_callback(self):
        msg = Odometry()
        msg.header.stamp = (self.get_clock().now() - Duration(seconds=0.01)).to_msg()
        msg.pose.pose.position.x = 1.0
        msg.pose.pose.position.y = 1.0
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0
        self.odom_publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FakeOdomPublisher()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()