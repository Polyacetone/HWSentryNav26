import rclpy
from rclpy.node import Node
from interfaces.msg import JointState
from sensor_msgs.msg import Imu

class FakeSerialPublisher(Node):
    def __init__(self):
        super().__init__('fake_serial_publisher')
        self.joint_state_publisher_ = self.create_publisher(JointState, '/serial_bridge/joint_state', 1)
        self.imu_publisher_ = self.create_publisher(Imu, '/serial_bridge/imu', 1)
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.get_logger().info('FakeSerialPublisher started.')
        # 新增变量用于角度变化
        self._t = 0.0

    def timer_callback(self):
        import math
        # 时间步进
        self._t += 0.01
        # yaw_angle 在 -pi~pi 匀速旋转
        yaw_angle = math.fmod(self._t, 2 * math.pi)
        # pitch_angle 在 -pi/6 ~ pi/6 之间振荡
        pitch_angle = (math.pi / 6) * math.sin(self._t)

        # 发布关节状态
        msg = JointState()
        msg.stamp = self.get_clock().now().to_msg()
        msg.pitch_angle = pitch_angle
        msg.yaw_angle = yaw_angle
        self.joint_state_publisher_.publish(msg)

        # 发布IMU消息
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0
        msg.orientation.w = 1.0
        self.imu_publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FakeSerialPublisher()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()