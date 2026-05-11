import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped

class TF2LookupNode(Node):
    def __init__(self):
        super().__init__('tf2_lookup_node')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0, self.lookup_transform)

    def lookup_transform(self):
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                'map', 'imu_world', rclpy.time.Time()
            )
            self.get_logger().info(
                f'Found transform from imu_world to map: \n'
                f'translation={transform.transform.translation}\n'
                f'rotation={transform.transform.rotation}\n\n'
            )
        except tf2_ros.TransformException as ex:
            self.get_logger().error(f'Transform lookup failed: {ex}')

def main(args=None):
    rclpy.init(args=args)
    node = TF2LookupNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()