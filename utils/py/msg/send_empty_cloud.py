import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

class PublishEmptyPointCloud(Node):
    def __init__(self):
        super().__init__('publish_empty_pointcloud')
        self.publisher_ = self.create_publisher(PointCloud2, '/small_glim/registered_cloud', 10)
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PublishEmptyPointCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()