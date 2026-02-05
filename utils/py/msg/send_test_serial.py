import rclpy
from rclpy.node import Node
from interfaces.msg import ChassisCmd
from interfaces.msg import ShootCmd
from tf2_ros import Buffer, TransformListener, LookupException
import math

class ChassisCmdPublisher(Node):
    def __init__(self):
        super().__init__('chassis_cmd_publisher')
        self.chassis_cmd_publisher_ = self.create_publisher(ChassisCmd, '/path_follower/chassis_cmd', 1)
        self.shoot_cmd_publisher_ = self.create_publisher(ShootCmd, '/autoaim_controller/shoot_cmd', 1)
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Initialize TF2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def timer_callback(self):
        msg = ChassisCmd()

        try:
            # Lookup transform from imu_link to imu_world
            transform = self.tf_buffer.lookup_transform('imu_world', 'imu_link', rclpy.time.Time())

            # Extract the rotation matrix components
            q = transform.transform.rotation
            x = 1.0  # imu_link x-axis in its own frame
            y = 0.0
            z = 0.0

            # Convert quaternion to rotation matrix
            # Quaternion components
            qw, qx, qy, qz = q.w, q.x, q.y, q.z

            # Rotation matrix components
            R11 = 1 - 2 * (qy**2 + qz**2)
            R12 = 2 * (qx * qy - qz * qw)
            R13 = 2 * (qx * qz + qy * qw)

            R21 = 2 * (qx * qy + qz * qw)
            R22 = 1 - 2 * (qx**2 + qz**2)
            R23 = 2 * (qy * qz - qx * qw)

            R31 = 2 * (qx * qz - qy * qw)
            R32 = 2 * (qy * qz + qx * qw)
            R33 = 1 - 2 * (qx**2 + qy**2)

            # Transform imu_link x-axis to imu_world frame
            x_world = R11 * x + R12 * y + R13 * z
            y_world = R21 * x + R22 * y + R23 * z

            # Calculate the angle of the projection in the xy-plane
            theta = math.atan2(y_world, x_world)

            msg.theta = theta
        except LookupException as e:
            self.get_logger().warn(f'Could not transform imu_link to imu_world: {e}')
            msg.theta = 0.0  # Default value if transform fails

        msg.velocity = 0.0
        msg.omega = 0.0
        msg.step_up_ahead = False
        msg.step_down_ahead = False
        msg.slow_spin = False
        msg.fast_spin = False
        self.chassis_cmd_publisher_.publish(msg)
        self.get_logger().info(f'Published ChassisCmd message with theta: {msg.theta}')

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