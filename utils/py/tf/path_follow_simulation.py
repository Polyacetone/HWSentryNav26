import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from interfaces.msg import JointState, ChassisStatus, ChassisCmd
from sensor_msgs.msg import PointCloud2
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
        
        # Parameters
        self.dt = 0.01
        
        # State
        self.x = 2.0
        self.y = 5.0
        self.theta = 0.0
        self.v = 0.0
        self.omega = 0.0
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
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        self.get_logger().info("Simulation Node Started")

    def cmd_callback(self, msg):
        self.v = msg.velocity
        self.omega = msg.palstance
        self.last_recv_time = self.get_clock().now()

    def timer_callback(self):
        if self.get_clock().now() - self.last_recv_time > rclpy.duration.Duration(seconds=0.3):
            self.v = 0.0
            self.omega = 0.0

        # Update state
        self.x += self.v * math.cos(self.theta) * self.dt
        self.y += self.v * math.sin(self.theta) * self.dt
        self.theta += self.omega * self.dt
        
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

        cloud_msg = PointCloud2()
        cloud_msg.header.stamp = self.get_clock().now().to_msg()
        cloud_msg.header.frame_id = "odom"
        self.cloud_publisher.publish(cloud_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SimulationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
