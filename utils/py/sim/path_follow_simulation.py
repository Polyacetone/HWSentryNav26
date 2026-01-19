import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from interfaces.msg import JointState, ChassisStatus, ChassisCmd
from sensor_msgs.msg import PointCloud2, PointField
import struct
from tf2_ros import TransformBroadcaster
import math

def euler_to_quaternion(roll, pitch, yaw):
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return Quaternion(x=qx, y=qy, z=qz, w=qw)

class SimulationNode(Node):
    def __init__(self):
        super().__init__('simulation_node')
        
        # State
        self.x = 2.0
        self.y = 5.0
        self.theta = 0.0
        self.v = 0.0
        self.last_v = 0.0
        self.omega = 0.0
        self.last_omega = 0.0
        self.slow_spin = False
        self.fast_spin = False
        self.last_recv_time = self.get_clock().now()
        
        # Publishers
        self.joint_state_pub = self.create_publisher(JointState, '/serial_bridge/joint_state', 2)
        self.odom_pub = self.create_publisher(Odometry, '/small_glim/odometry', 2)
        self.imu_pub = self.create_publisher(Imu, '/serial_bridge/imu', 2)
        self.chassis_status_pub = self.create_publisher(ChassisStatus, '/serial/chassis_status', 2)
        self.cloud_publisher = self.create_publisher(PointCloud2, '/small_glim/registered_cloud', 2)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscribers
        self.create_subscription(ChassisCmd, '/path_follower/chassis_cmd', self.cmd_callback, 2)
        
        # Timer
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.timer_count = 0
        
        self.get_logger().info("Simulation Node Started")

    def cmd_callback(self, msg):
        self.last_v = self.v
        self.last_omega = self.omega
        self.v = msg.velocity
        self.omega = msg.palstance
        self.slow_spin = msg.slow_spin
        self.fast_spin = msg.fast_spin
        if self.fast_spin or self.slow_spin: # 小陀螺模式
            # 强制设定角速度和线速度
            if self.fast_spin: self.omega = 12.5 * (1 if self.last_omega >= 0 else -1)
            else: self.omega = 6.0 * (1 if self.last_omega >= 0 else -1)
            # 限制加速度
            if abs(self.last_v) / 0.1 > 3.0:
                print("High acceleration detected before spin: ", abs(self.last_v) / 0.1)
                self.v = 0.1 * (3.0 * (1 if self.last_v > 0 else -1))
            else: self.v = 0.0
            # 限制角加速度
            if abs(self.omega - self.last_omega) / 0.1 > 12.5:
                self.omega = 0.1 * (12.5 * (1 if self.omega - self.last_omega > 0 else -1)) + self.last_omega
        else: # 普通模式
            if abs(self.v - self.last_v) / 0.1 > 3.0: # 限制加速度
                print("High acceleration detected: ", abs(self.v - self.last_v) / 0.1)
                self.v = 0.1 * (3.0 * (1 if self.v - self.last_v > 0 else -1)) + self.last_v
            if abs(self.omega - self.last_omega) / 0.1 > 12.5: # 限制角加速度
                print("High angular acceleration detected: ", abs(self.omega - self.last_omega) / 0.1)
                self.omega = 0.1 * (12.5 * (1 if self.omega - self.last_omega > 0 else -1)) + self.last_omega
            if abs(self.v - self.last_v) / 0.1 * abs(self.omega - self.last_omega) / 0.1 > 3.5: # 限制线速度和角速度乘积
                print("High combined acceleration detected: ", abs(self.v - self.last_v) / 0.1 * abs(self.omega - self.last_omega) / 0.1)
        self.last_recv_time = self.get_clock().now()

    def timer_callback(self):
        self.timer_count += 1
        if self.get_clock().now() - self.last_recv_time > rclpy.duration.Duration(seconds=0.3):
            self.v = 0.0
            self.omega = 0.0
            self.last_v = 0.0
            self.last_omega = 0.0

        # Update state
        self.x += self.v * math.cos(self.theta) * 0.01
        self.y += self.v * math.sin(self.theta) * 0.01
        self.theta += self.omega * 0.01
        
        # Publish Imu (imu_link in imu_world)
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = "imu_world"
        imu_msg.orientation = euler_to_quaternion(0, 0, self.theta)
        self.imu_pub.publish(imu_msg)
        
        # Publish JointState (Fixed)
        joint_msg = JointState()
        joint_msg.stamp = self.get_clock().now().to_msg()
        joint_msg.yaw_angle = 0.0
        joint_msg.pitch_angle = 0.0
        self.joint_state_pub.publish(joint_msg)
        
        # Publish ChassisStatus
        status_msg = ChassisStatus()
        status_msg.velocity = self.v
        status_msg.palstance = self.omega
        status_msg.leg_mode = 4 # Mature mode
        self.chassis_status_pub.publish(status_msg)
        
        # Publish map -> odom TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        self.tf_broadcaster.sendTransform(t)

        # Publish Odometry (imu_link in odom)
        odom_msg = Odometry()
        odom_msg.header.stamp = (self.get_clock().now() - rclpy.duration.Duration(seconds=0.03)).to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "imu_link" # As expected by tf_maintainer logic
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = euler_to_quaternion(0, 0, self.theta)
        self.odom_pub.publish(odom_msg)

        # Publish PointCloud2 (Every 10 cycles)
        if self.timer_count % 10 == 0:
            # moving sphere parameters
            radius = 0.2
            cx = 14.0
            cz = 1.0
            # time in seconds (timer runs at 0.01s)
            t = self.timer_count * 0.01
            period = 10.0
            cy = 5.5 + 0.5 * math.sin(2.0 * math.pi * t / period)

            # sample sphere surface with grid in spherical coordinates
            num_theta = 20
            num_phi = 20
            points = []
            for i in range(num_phi):
                phi = math.pi * (i + 0.5) / num_phi
                for j in range(num_theta):
                    theta = 2.0 * math.pi * j / num_theta
                    px = cx + radius * math.sin(phi) * math.cos(theta)
                    py = cy + radius * math.sin(phi) * math.sin(theta)
                    pz = cz + radius * math.cos(phi)
                    points.append((px, py, pz))

            cloud_msg = PointCloud2()
            cloud_msg.header.stamp = self.get_clock().now().to_msg()
            cloud_msg.header.frame_id = "odom"
            cloud_msg.height = 1
            cloud_msg.width = len(points)

            # fields: x, y, z as float32
            cloud_msg.fields = []
            cloud_msg.fields.append(PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1))
            cloud_msg.fields.append(PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1))
            cloud_msg.fields.append(PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1))

            cloud_msg.is_bigendian = False
            cloud_msg.point_step = 12  # 3 * 4 bytes
            cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
            cloud_msg.is_dense = True

            # pack points as float32 little-endian
            data = bytearray()
            for (px, py, pz) in points:
                data += struct.pack('<fff', float(px), float(py), float(pz))

            cloud_msg.data = bytes(data)
            self.cloud_publisher.publish(cloud_msg)
    
def main(args=None):
    rclpy.init(args=args)
    node = SimulationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
