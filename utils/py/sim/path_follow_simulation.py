import math
import random
import struct
from collections import deque
from dataclasses import dataclass

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

from geometry_msgs.msg import Quaternion, TransformStamped
from interfaces.msg import ChassisCmd, ChassisStatus, JointState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2, PointField
from tf2_ros import TransformBroadcaster

def euler_to_quaternion(roll, pitch, yaw):
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return Quaternion(x=qx, y=qy, z=qz, w=qw)

@dataclass
class ControlCommand:
    stamp: rclpy.time.Time
    velocity: float
    palstance: float
    slow_spin: bool
    fast_spin: bool

@dataclass
class ScalarCommand:
    stamp: rclpy.time.Time
    value: float
    slow_spin: bool
    fast_spin: bool

class SimulationNode(Node):
    def __init__(self):
        super().__init__('simulation_node')

        # Parameters (make simulation less ideal)
        self.declare_parameter('timer_period_sec', 0.01)
        self.declare_parameter('cmd_timeout_sec', 0.3)
        # Separate delays for linear and angular control
        self.declare_parameter('control_delay_v_sec', 0.03)
        self.declare_parameter('control_delay_w_sec', 0.03)

        # Angular channel: first-order lag (closed-loop tracking dynamics)
        # After transport delay, the effective commanded target is tracked with a first-order lag.
        # Smaller tau -> faster tracking.
        self.declare_parameter('lag_tau_w_sec', 0.2)

        # Linear channel: second-order surrogate model (inverted pendulum under LQR)
        #   v_dot = a
        #   a_dot = (1/tau) * ( K*(vcmd - v) - a )
        # where a is an "agent acceleration" state (pitch-equivalent surrogate).
        # tau is the pitch build-up time constant you observed (~1s scale).
        # K is the LQR sensitivity from velocity error to pitch command.
        self.declare_parameter('linear_tau_sec', 0.2)
        self.declare_parameter('linear_gain_k', 4.0)

        self.declare_parameter('velocity_noise_std', 0.02)  # m/s
        self.declare_parameter('palstance_noise_std', 0.02)  # rad/s
        self.declare_parameter('accel_limit', 3.0)  # m/s^2
        self.declare_parameter('ang_accel_limit', 12.5)  # rad/s^2
        self.declare_parameter('spin_fast_omega', 12.5)  # rad/s
        self.declare_parameter('spin_slow_omega', 6.0)  # rad/s
        self.declare_parameter('odom_time_offset_sec', 0.03)
        self.declare_parameter('random_seed', 0)

        timer_period_sec = float(self.get_parameter('timer_period_sec').value)
        self._cmd_timeout = Duration(seconds=float(self.get_parameter('cmd_timeout_sec').value))
        self._control_delay_v = Duration(seconds=float(self.get_parameter('control_delay_v_sec').value))
        self._control_delay_w = Duration(seconds=float(self.get_parameter('control_delay_w_sec').value))

        self._lag_tau_w = float(self.get_parameter('lag_tau_w_sec').value)

        self._linear_tau = float(self.get_parameter('linear_tau_sec').value)
        self._linear_k = float(self.get_parameter('linear_gain_k').value)

        self._v_noise_std = float(self.get_parameter('velocity_noise_std').value)
        self._w_noise_std = float(self.get_parameter('palstance_noise_std').value)
        self._accel_limit = float(self.get_parameter('accel_limit').value)
        self._ang_accel_limit = float(self.get_parameter('ang_accel_limit').value)
        self._spin_fast_omega = float(self.get_parameter('spin_fast_omega').value)
        self._spin_slow_omega = float(self.get_parameter('spin_slow_omega').value)
        self._odom_time_offset = Duration(seconds=float(self.get_parameter('odom_time_offset_sec').value))

        seed = int(self.get_parameter('random_seed').value)
        if seed != 0:
            random.seed(seed)

        # State
        self.x = 2.0
        self.y = 5.0
        self.theta = 0.0
        self.v = 0.0
        self.a = 0.0
        self.omega = 0.0

        # Angular target is tracked through a first-order lag.
        self._w_lagged = 0.0

        # Control pipeline: recv -> (v-delay queue, w-delay queue) -> noisy target -> rate-limited actual
        self._v_cmd_queue: deque[ScalarCommand] = deque(maxlen=500)
        self._w_cmd_queue: deque[ScalarCommand] = deque(maxlen=500)
        self._last_cmd_recv_time = self.get_clock().now()
        self._applied_v_cmd: ScalarCommand | None = None
        self._applied_w_cmd: ScalarCommand | None = None

        # Publishers
        self.joint_state_pub = self.create_publisher(JointState, '/serial_bridge/joint_state', 2)
        self.odom_pub = self.create_publisher(Odometry, '/small_glim/odometry', 2)
        self.imu_pub = self.create_publisher(Imu, '/serial_bridge/imu', 2)
        self.chassis_status_pub = self.create_publisher(ChassisStatus, '/serial_bridge/chassis_status', 2)
        self.cloud_publisher = self.create_publisher(PointCloud2, '/small_glim/registered_cloud', 2)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribers
        self.create_subscription(ChassisCmd, '/path_follower/chassis_cmd', self.cmd_callback, 2)

        # Timer
        self._dt = timer_period_sec
        self.timer = self.create_timer(self._dt, self.timer_callback)
        self.timer_count = 0

        self.get_logger().info(
            "Simulation Node Started (delay_v=%.3fs, delay_w=%.3fs, linear_tau=%.3fs, linear_k=%.3f, tau_w=%.3fs, noise_v_std=%.3f, noise_w_std=%.3f)"
            % (
                self._control_delay_v.nanoseconds * 1e-9,
                self._control_delay_w.nanoseconds * 1e-9,
                self._linear_tau,
                self._linear_k,
                self._lag_tau_w,
                self._v_noise_std,
                self._w_noise_std,
            )
        )

    def cmd_callback(self, msg):
        now = self.get_clock().now()
        self._last_cmd_recv_time = now
        slow_spin = bool(msg.slow_spin)
        fast_spin = bool(msg.fast_spin)
        self._v_cmd_queue.append(
            ScalarCommand(
                stamp=now,
                value=float(msg.velocity),
                slow_spin=slow_spin,
                fast_spin=fast_spin,
            )
        )
        self._w_cmd_queue.append(
            ScalarCommand(
                stamp=now,
                value=float(msg.palstance),
                slow_spin=slow_spin,
                fast_spin=fast_spin,
            )
        )

    @staticmethod
    def _pop_delayed_scalar(
        queue: deque[ScalarCommand],
        now: rclpy.time.Time,
        delay: Duration,
    ) -> ScalarCommand | None:
        """Return newest scalar command that has waited at least delay."""
        if not queue:
            return None

        ready_time = now - delay
        chosen: ScalarCommand | None = None
        while queue and queue[0].stamp <= ready_time:
            chosen = queue.popleft()
        return chosen

    def _compute_target_v(self, cmd: ScalarCommand | None) -> float:
        if cmd is None:
            target_v = 0.0
        else:
            target_v = cmd.value
            if cmd.fast_spin or cmd.slow_spin:
                target_v = 0.0

        if self._v_noise_std > 0.0:
            target_v += random.gauss(0.0, self._v_noise_std)
        return target_v

    def _compute_target_w(self, cmd: ScalarCommand | None) -> float:
        if cmd is None:
            target_w = 0.0
        else:
            target_w = cmd.value
            if cmd.fast_spin or cmd.slow_spin:
                last_sign = 1.0 if self.omega >= 0.0 else -1.0
                spin_w = self._spin_fast_omega if cmd.fast_spin else self._spin_slow_omega
                target_w = spin_w * last_sign

        if self._w_noise_std > 0.0:
            target_w += random.gauss(0.0, self._w_noise_std)
        return target_w

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _first_order_lag(current: float, target: float, tau: float, dt: float) -> float:
        """First-order lag: y[k+1] = alpha*y[k] + (1-alpha)*u[k], alpha=exp(-dt/tau)."""
        if dt <= 0.0:
            return current
        if tau <= 0.0:
            return target
        tau_safe = max(1e-4, tau)
        alpha = math.exp(-dt / tau_safe)
        return alpha * current + (1.0 - alpha) * target

    def _rate_limit_omega(self, target_w: float) -> None:
        """Move actual omega toward target with angular acceleration limits."""
        dt = self._dt
        if dt <= 0.0:
            return
        dw_max = self._ang_accel_limit * dt
        dw = self._clamp(target_w - self.omega, -dw_max, dw_max)
        self.omega += dw

    def _update_linear_second_order(self, v_cmd: float) -> None:
        """Second-order linear dynamics with surrogate acceleration state.

        Continuous model:
            v_dot = a
            a_dot = (1/tau) * ( K*(v_cmd - v) - a )

        Discretization: explicit Euler. Also applies accel saturation via `accel_limit`.
        """
        dt = self._dt
        if dt <= 0.0:
            return

        # tau<=0 -> instantaneous a follows K*(error)
        if self._linear_tau <= 0.0:
            a_target = self._linear_k * (v_cmd - self.v)
            self.a = self._clamp(a_target, -self._accel_limit, self._accel_limit)
        else:
            inv_tau = 1.0 / max(1e-4, self._linear_tau)
            a_dot = inv_tau * (self._linear_k * (v_cmd - self.v) - self.a)
            self.a += a_dot * dt
            self.a = self._clamp(self.a, -self._accel_limit, self._accel_limit)

        self.v += self.a * dt

    def timer_callback(self):
        self.timer_count += 1
        now = self.get_clock().now()

        # If command stream is lost, stop (also clear queued delayed commands)
        if now - self._last_cmd_recv_time > self._cmd_timeout:
            self._v_cmd_queue.clear()
            self._w_cmd_queue.clear()
            self._applied_v_cmd = None
            self._applied_w_cmd = None

        # Apply delayed control (hold last applied scalar cmd if no new delayed cmd ready)
        new_v_cmd = self._pop_delayed_scalar(self._v_cmd_queue, now, self._control_delay_v)
        if new_v_cmd is not None:
            self._applied_v_cmd = new_v_cmd

        new_w_cmd = self._pop_delayed_scalar(self._w_cmd_queue, now, self._control_delay_w)
        if new_w_cmd is not None:
            self._applied_w_cmd = new_w_cmd

        target_v_cmd = self._compute_target_v(self._applied_v_cmd)
        target_w = self._compute_target_w(self._applied_w_cmd)

        # Linear channel: second-order model after transport delay (+noise, +spin gating)
        self._update_linear_second_order(target_v_cmd)

        # Angular channel: first-order lag + angular acceleration limits
        self._w_lagged = self._first_order_lag(self._w_lagged, target_w, self._lag_tau_w, self._dt)
        self._rate_limit_omega(self._w_lagged)

        # Update state using actual velocity
        dt = self._dt
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        self.theta += self.omega * dt

        # Publish Imu (imu_link in imu_world)
        imu_msg = Imu()
        imu_msg.header.stamp = now.to_msg()
        imu_msg.header.frame_id = "imu_world"
        imu_msg.orientation = euler_to_quaternion(0, 0, self.theta)
        self.imu_pub.publish(imu_msg)

        # Publish JointState (Fixed)
        joint_msg = JointState()
        joint_msg.stamp = now.to_msg()
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
        tf_msg = TransformStamped()
        tf_msg.header.stamp = now.to_msg()
        tf_msg.header.frame_id = "map"
        tf_msg.child_frame_id = "odom"
        tf_msg.transform.translation.x = 0.0
        tf_msg.transform.translation.y = 0.0
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.w = 1.0
        tf_msg.transform.rotation.x = 0.0
        tf_msg.transform.rotation.y = 0.0
        tf_msg.transform.rotation.z = 0.0
        self.tf_broadcaster.sendTransform(tf_msg)

        # Publish Odometry (imu_link in odom)
        odom_msg = Odometry()
        odom_msg.header.stamp = (now - self._odom_time_offset).to_msg()
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
            cx = 6.5
            cz = 0.3
            # time in seconds
            sim_time_sec = self.timer_count * self._dt
            period = 10.0
            cy = 6.0 + 0.5 * math.sin(2.0 * math.pi * sim_time_sec / period)

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
            cloud_msg.header.stamp = now.to_msg()
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
