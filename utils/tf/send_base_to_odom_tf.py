import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time

class BaseToOdomTFPublisher(Node):
    def __init__(self):
        super().__init__('base_to_odom_tf_publisher')
        self.br = TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.publish_transform)  # 100ms间隔
        self.get_logger().info("Publishing transform from base to odom every 100ms")

    def publish_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base'
      
        t.transform.translation.x = 3.0
        t.transform.translation.y = 4.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
      
        self.br.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = BaseToOdomTFPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()