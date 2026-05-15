import math
from collections import deque
from dataclasses import dataclass
import struct
from typing import Optional

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

from geometry_msgs.msg import Quaternion, TransformStamped
from interfaces.msg import ChassisCmd, ChassisStatus, CostMaps, JointState
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Image, Imu, PointCloud2, PointField
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))

# =============================================================================
# 功率模型（与 C++ MPC 中的系数一致）
# =============================================================================

_PWR_C = (
    3.1183599570e+00,  # c0: 1 (bias)
    3.4172476463e+01,  # c1: v·a
    1.0359111933e+00,  # c2: ω·α
    3.6371494354e+00,  # c3: a²
    2.3486803448e-02,  # c4: α²
    2.7300289323e+01,  # c5: |v|
    2.6315570711e+00,  # c6: |ω|
    1.8359691253e+00,  # c7: v²
    1.1200532785e+00,  # c8: ω²
    2.6043584920e-01,  # c9: |a|
    5.2574769643e-02, # c10: |α|
    0.0000000000e+00  # c11: |v·ω|
)
_PWR_EPS2 = 0.05 ** 2

def predict_chassis_power(v: float, w: float, a: float, alpha: float) -> float:
    """Predict chassis electrical power [W] from the identified model."""
    def _sa(x: float) -> float:
        return math.sqrt(x * x + _PWR_EPS2)
    c = _PWR_C
    return (c[0]
            + c[1] * v * a      + c[2] * w * alpha
            + c[3] * a * a      + c[4] * alpha * alpha
            + c[5] * _sa(v)     + c[6] * _sa(w)
            + c[7] * v * v      + c[8] * w * w
            + c[9] * _sa(a)     + c[10] * _sa(alpha)
            + c[11] * _sa(v * w))

def angle_diff(a: float, b: float) -> float:
    return wrap_to_pi(a - b)

def is_finite(x: float) -> bool:
    return math.isfinite(x)

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
    qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
    qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
    qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
    return Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))

# =============================================================================
# 动态障碍物规格（在 SimConfig 中引用，需先定义）
# =============================================================================


@dataclass
class ObstacleSpec:
    """单个动态障碍物的运动参数规格。

    motion_type : "circle"  圆形轨迹（围绕随机中心做匀速/变速圆周运动）
                  "line"    直线往返（在随机方向的线段上来回运动，端点处平滑换向）
    speed_type  : "constant"    匀速（恒定 max_speed）
                  "oscillating" 变速（速度按正弦规律在 0 ~ max_speed 之间振荡）
    """
    motion_type: str = "circle"
    speed_type: str = "constant"
    max_speed: float = 0.8             # 最大线速度 (m/s)
    max_accel: float = 1.5             # 最大加速度 (m/s²)，用于速度过渡 & 直线端点制动
    circle_radius: float = 2.0         # 圆形轨迹轨道半径 (m)，仅 circle 生效
    line_length: float = 4.0           # 直线往返总长度 (m)，仅 line 生效
    oscillate_freq: float = 0.25       # 变速振荡频率 (Hz)，仅 oscillating 生效


# =============================================================================
# 配置区（统一写在最上面，直接改即可）
# =============================================================================


@dataclass
class SimConfig:
    # --- 频率 ---
    LQR_FREQ_HZ: float = 1000.0
    IMU_PUB_HZ: float = 100.0
    JOINT_PUB_HZ: float = 20.0
    STATUS_PUB_HZ: float = 20.0
    ODOM_PUB_HZ: float = 20.0
    CLOUD_PUB_HZ: float = 20.0
    TF_PUB_HZ: float = 1.0

    # --- 超时 ---
    CMD_TIMEOUT_SEC: float = 0.3

    # --- 初始位姿（世界坐标） ---
    INIT_X: float = 3.0
    INIT_Y: float = 3.5
    INIT_THETA: float = 0.0

    # --- 指令限幅（目标值） ---
    MAX_VELOCITY: float = 2.8
    MAX_PALSTANCE: float = 7.2
    MAX_ACCEL: float = 2.5
    MAX_ANG_ACCEL: float = 12.0
    MAX_V_W_PRODUCT: float = 3.5

    # --- 小陀螺 ---
    SPIN_FAST_OMEGA: float = 7.2
    SPIN_SLOW_OMEGA: float = 3.0

    # --- 里程计延迟 ---
    ODOM_DELAY_SEC: float = 0.05

    # --- 传感器噪声（标准差）---
    IMU_YAW_NOISE_STD_RAD: float = 0.01

    ODOM_X_NOISE_STD_M: float = 0.004
    ODOM_Y_NOISE_STD_M: float = 0.004
    ODOM_YAW_NOISE_STD_RAD: float = 0.01
    ODOM_V_NOISE_STD_MPS: float = 0.01
    ODOM_W_NOISE_STD_RADPS: float = 0.01

    # --- 力矩扰动（每个通道）---
    # u = [T_w_l, T_w_r, T_b_l, T_b_r]
    TORQUE_BIAS_NM: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    TORQUE_NOISE_STD_NM: tuple[float, float, float, float] = (0.04, 0.04, 0.04, 0.04)

    # --- 话题名 ---
    TOPIC_CMD: str = "/path_follower/chassis_cmd"
    TOPIC_JOINT: str = "/serial_bridge/joint_state"
    TOPIC_ODOM: str = "/small_glim/odometry"
    TOPIC_IMU: str = "/serial_bridge/imu_pose"
    TOPIC_STATUS: str = "/serial_bridge/chassis_status"
    TOPIC_CLOUD: str = "/small_glim/registered_cloud"

    # --- 地图话题（参考 path_follower/path_planner/map_server）---
    TOPIC_GLOBAL_COST_MAP: str = "/map_server/global_cost_map"
    TOPIC_LOCAL_COST_MAPS: str = "/map_server/local_cost_maps"
    TOPIC_GLOBAL_DIRECTION_MAP: str = "/map_server/global_direction_map"

    # --- 障碍物判定（cost map）---
    # cost map 值域按工程惯例认为是 0~255（OccupancyGrid 的 int8 通过 uint8 解释）
    OBSTACLE_COST_THRESHOLD: int = 200  # >= 该值认为是障碍物/发生接触
    OBSTACLE_PUSH_COST_THRESHOLD: int = 250  # 当前位置 cost >= 该值则被“挤开”
    OBSTACLE_PUSH_SPEED_MPS: float = 0.6  # 被推开速度（m/s）
    ROBOT_INSCRIBED_RADIUS_M: float = 0.30  # 机器人内切圆半径（用于碰撞点采样）
    COLLISION_LOOKAHEAD_M: float = 0.05  # 前/后碰撞点额外外扩（m）

    # --- 台阶判定（direction map）---
    STEP_NORM_THRESHOLD: float = 0.9  # direction 向量模长 >= 该值认为在台阶区域
    STEP_STUCK_HEADING_ERR_RAD: float = math.radians(30.0)  # 与台阶方向场夹角过大则卡住
    STEP_STUCK_MIN_SPEED_MPS: float = 0.4  # 速度过小则卡住（仅在尝试前进时）
    STEP_RELEASE_NORM_THRESHOLD: float = 0.6  # 离开台阶区域的判定阈值（建议略小于进入阈值）

    # --- 小陀螺漂移 ---
    SPIN_DRIFT_SPEED_MPS: float = 0.02  # 小陀螺位置漂移速度（m/s）
    SPIN_DRIFT_DIR_X: float = 1.0
    SPIN_DRIFT_DIR_Y: float = 0.0

    # --- 功率/能量仿真 ---
    RFR_PWR_LIMIT: float = 80.0           # 裁判系统最大取电功率 (W)
    CAPACITOR_MAX_ENERGY: float = 1300.0   # 电容最大可用电量 (J)
    CAPACITOR_INIT_ENERGY: float = 1300.0  # 初始电容电量 (J)
    POWER_LOWPASS_HZ: float = 3.0         # 功率模型速度低通截频 (Hz)

    # --- 动态障碍物 ---
    # 障碍物生成范围（世界坐标，单位 m）
    OBSTACLE_SPAWN_X_MIN: float = 0.0
    OBSTACLE_SPAWN_X_MAX: float = 25.0
    OBSTACLE_SPAWN_Y_MIN: float = 0.0
    OBSTACLE_SPAWN_Y_MAX: float = 16.0

    # 障碍物点云外形（每个障碍物为球形点云，球心在 xy 平面运动，z 固定）
    OBSTACLE_CLOUD_RADIUS_M: float = 0.30          # 球半径 (m)
    OBSTACLE_CLOUD_CENTER_Z_M: float = 0.50        # 球心固定 z 高度 (m)
    OBSTACLE_CLOUD_DENSITY_PTS_PER_M2: float = 120.0  # 球面点密度 (points/m^2)
    OBSTACLE_CLOUD_MIN_POINTS: int = 80            # 每个障碍物最少点数

    # 障碍物种类列表（每项为 ObstacleSpec 实例，可自由增删）
    OBSTACLE_SPECS: tuple = (
        # ObstacleSpec(motion_type="circle",  speed_type="constant",    max_speed=1.0,  max_accel=1.2, circle_radius=2.5),
        # ObstacleSpec(motion_type="circle",  speed_type="constant",    max_speed=1.5,  max_accel=1.8, circle_radius=2.2),
        # ObstacleSpec(motion_type="circle",  speed_type="oscillating", max_speed=2.0,  max_accel=1.5, circle_radius=1.8, oscillate_freq=0.20),
        # ObstacleSpec(motion_type="line",    speed_type="constant",    max_speed=1.0,  max_accel=1.5, line_length=6.0),
        # ObstacleSpec(motion_type="line",    speed_type="constant",    max_speed=1.5,  max_accel=1.2, line_length=3.0),
        # ObstacleSpec(motion_type="line",    speed_type="oscillating", max_speed=2.0,  max_accel=2.0, line_length=5.0,   oscillate_freq=0.30),
    )

    # --- Frame 约定 ---
    FRAME_IMU_WORLD: str = "imu_world"
    FRAME_ODOM: str = "odom"
    FRAME_IMU_LINK: str = "imu_link"


# =============================================================================
# 机器人参数 + LQR 多项式（与原始模型一致）
# =============================================================================


@dataclass
class RobotParams:
    Ts: float = 0.001

    R_w: float = 0.12 / 2
    R_l: float = 0.442 / 2

    l_l: float = 0.15
    l_r: float = 0.15

    l_c: float = 0.04275

    m_w: float = 0.31509
    m_l: float = 3.38608
    m_b: float = 9.59766

    I_w: float = 0.0050935
    I_l_l: float = 0.02155
    I_l_r: float = 0.02155
    I_b: float = 1.0429
    I_z: float = 0.495599

    g: float = 9.7936

    h_min: float = 0.10
    h_max: float = 0.33

    T_w_max: float = 20.0
    T_b_max: float = 30.0


class PolynomialFit:
    """随腿长变化的 LQR 增益 K(l_l,l_r) 多项式。"""

    # 说明：K_LQR_POLY[i,j] 的长度应为 10（poly33 系数）
    # LQR增益多项式系数 [4][10][10]
    # poly33: p00 + p10*x + p01*y + p20*x^2 + p11*x*y + p02*y^2 + p30*x^3 + p21*x^2*y + p12*x*y^2 + p03*y^3
    K_LQR_POLY = np.array([
        [
            [-1.628477,0.074972,-0.337619,-0.025388,0.015035,0.069129,0.007960,-0.002058,-0.004120,-0.005938],
            [-3.148457,0.121836,-0.644343,-0.048678,0.026057,0.122704,0.015100,-0.003736,-0.006535,-0.010440],
            [-5.299122,-0.482103,0.944943,0.039523,-0.012309,-0.074288,0.003024,-0.005601,0.011094,-0.010894],
            [-1.244501,-0.149874,0.254599,0.014269,-0.007122,-0.011812,-0.000639,-0.000488,0.003845,-0.003717],
            [-9.003710,0.704055,-2.694577,-0.118707,0.149293,0.101943,0.026308,-0.021750,-0.026247,0.024070],
            [-1.203166,0.045088,-0.407403,-0.010402,0.016968,-0.010755,0.002693,-0.003772,-0.000176,0.004082],
            [-6.596674,-1.227503,0.078447,0.069823,-0.034993,0.135025,-0.001189,-0.006023,0.028164,-0.026930],
            [-0.682517,-0.191308,-0.065611,-0.004367,-0.022336,0.020326,0.001991,-0.000411,0.005345,-0.002601],
            [-10.197844,0.988028,0.922401,-0.064750,-0.151307,0.118259,-0.000414,0.021426,-0.010580,-0.059034],
            [-1.674121,0.173599,0.105718,-0.019740,-0.022532,0.023995,0.002861,0.003555,-0.001900,-0.008281],
        ],
        [
            [-1.628477,-0.337619,0.074972,0.069129,0.015035,-0.025388,-0.005938,-0.004120,-0.002058,0.007960],
            [-3.148457,-0.644343,0.121836,0.122704,0.026057,-0.048678,-0.010440,-0.006535,-0.003736,0.015100],
            [5.299122,-0.944943,0.482103,0.074288,0.012309,-0.039523,0.010894,-0.011094,0.005601,-0.003024],
            [1.244501,-0.254599,0.149874,0.011812,0.007122,-0.014269,0.003717,-0.003845,0.000488,0.000639],
            [-6.596674,0.078447,-1.227503,0.135025,-0.034993,0.069823,-0.026930,0.028164,-0.006023,-0.001189],
            [-0.682517,-0.065611,-0.191308,0.020326,-0.022336,-0.004367,-0.002601,0.005345,-0.000411,0.001991],
            [-9.003710,-2.694577,0.704055,0.101943,0.149293,-0.118707,0.024070,-0.026247,-0.021750,0.026308],
            [-1.203166,-0.407403,0.045088,-0.010755,0.016968,-0.010402,0.004082,-0.000176,-0.003772,0.002693],
            [-10.197844,0.922401,0.988028,0.118259,-0.151307,-0.064750,-0.059034,-0.010580,0.021426,-0.000414],
            [-1.674121,0.105718,0.173599,0.023995,-0.022532,-0.019740,-0.008281,-0.001900,0.003555,0.002861],
        ],
        [
            [10.052848,-0.592304,-1.337429,0.285374,0.153680,-0.393905,-0.065216,-0.034707,-0.019904,0.190034],
            [18.641363,-1.019835,-2.383568,0.526303,0.273279,-0.716587,-0.120320,-0.064579,-0.043178,0.344894],
            [-32.502883,-1.097253,-2.708689,0.300917,-0.471990,0.986675,-0.029189,0.061210,-0.005253,-0.138001],
            [-7.206430,-0.216004,-0.895618,0.036942,-0.135131,0.221764,0.003431,0.013254,-0.011105,-0.017681],
            [46.924361,-0.791132,4.180590,0.638575,0.576064,-1.849876,-0.191294,-0.008540,0.123274,0.320496],
            [6.008345,-0.114179,0.298496,0.079248,0.006594,-0.126514,-0.020529,0.015774,0.009673,0.015723],
            [4.649902,0.079766,-6.495679,0.314894,-0.812332,-0.257809,0.006652,0.062674,-0.188647,0.411908],
            [2.126687,0.657304,-0.613939,0.036681,-0.092386,-0.097412,-0.003321,0.004217,-0.041474,0.060645],
            [-71.249707,2.611689,-15.012199,-0.156869,1.106750,1.843608,-0.105687,-0.146069,-0.149076,0.193295],
            [-6.268344,0.168969,-2.054454,0.056590,0.200939,0.150708,-0.030971,-0.034931,-0.026320,0.059490],
        ],
        [
            [10.052848,-1.337429,-0.592304,-0.393905,0.153680,0.285374,0.190034,-0.019904,-0.034707,-0.065216],
            [18.641363,-2.383568,-1.019835,-0.716587,0.273279,0.526303,0.344894,-0.043178,-0.064579,-0.120320],
            [32.502883,2.708689,1.097253,-0.986675,0.471990,-0.300917,0.138001,0.005253,-0.061210,0.029189],
            [7.206430,0.895618,0.216004,-0.221764,0.135131,-0.036942,0.017681,0.011105,-0.013254,-0.003431],
            [4.649902,-6.495679,0.079766,-0.257809,-0.812332,0.314894,0.411908,-0.188647,0.062674,0.006652],
            [2.126687,-0.613939,0.657304,-0.097412,-0.092386,0.036681,0.060645,-0.041474,0.004217,-0.003321],
            [46.924361,4.180590,-0.791132,-1.849876,0.576064,0.638575,0.320496,0.123274,-0.008540,-0.191294],
            [6.008345,0.298496,-0.114179,-0.126514,0.006594,0.079248,0.015723,0.009673,0.015774,-0.020529],
            [-71.249707,-15.012199,2.611689,1.843608,1.106750,-0.156869,0.193295,-0.149076,-0.146069,-0.105687],
            [-6.268344,-2.054454,0.168969,0.150708,0.200939,0.056590,0.059490,-0.026320,-0.034931,-0.030971],
        ],
    ])

    @staticmethod
    def eval_poly33(coeffs: np.ndarray, x: float, y: float) -> float:
        return (
            coeffs[0]
            + coeffs[1] * x
            + coeffs[2] * y
            + coeffs[3] * x**2
            + coeffs[4] * x * y
            + coeffs[5] * y**2
            + coeffs[6] * x**3
            + coeffs[7] * x**2 * y
            + coeffs[8] * x * y**2
            + coeffs[9] * y**3
        )

    # 前馈力矩多项式系数 u_d(l_l, l_r)  [4][10]
    U_D_POLY = np.array([
        [-0.356311,0.039795,0.039795,0.006540,-0.018493,0.006540,-0.001946,0.001525,0.001525,-0.001946],
        [-0.356311,0.039795,0.039795,0.006540,-0.018493,0.006540,-0.001946,0.001525,0.001525,-0.001946],
        [-0.708068,-0.025755,-0.025755,0.010758,-0.011525,0.010758,-0.000669,-0.000140,-0.000140,-0.000669],
        [-0.708068,-0.025755,-0.025755,0.010758,-0.011525,0.010758,-0.000669,-0.000140,-0.000140,-0.000669],
    ])

    @classmethod
    def get_K_matrix(cls, l_l: float, l_r: float) -> np.ndarray:
        K = np.zeros((4, 10))
        for i in range(4):
            for j in range(10):
                K[i, j] = cls.eval_poly33(cls.K_LQR_POLY[i, j], l_l, l_r)
        return K

    @classmethod
    def get_u_d(cls, l_l: float, l_r: float) -> np.ndarray:
        """计算前馈力矩 u_d(l_l, l_r)，对应平衡点处的控制输入。"""
        u_d = np.zeros(4)
        for i in range(4):
            u_d[i] = cls.eval_poly33(cls.U_D_POLY[i], l_l, l_r)
        return u_d


# =============================================================================
# 动力学（复用原 A/B 多项式实现，保持一致）
# =============================================================================


class WheelLegDynamics:
    IDX_S = 0
    IDX_DS = 1
    IDX_PHI = 2
    IDX_DPHI = 3
    IDX_THETA_L_L = 4
    IDX_DTHETA_L_L = 5
    IDX_THETA_L_R = 6
    IDX_DTHETA_L_R = 7
    IDX_THETA_B = 8
    IDX_DTHETA_B = 9

    def __init__(self, params: Optional[RobotParams] = None):
        self.params = params if params is not None else RobotParams()
        self.state = np.zeros(10)
        self.world_pose = np.zeros(3)

        self._cached_l_l: Optional[float] = None
        self._cached_l_r: Optional[float] = None
        self._cached_A: Optional[np.ndarray] = None
        self._cached_B: Optional[np.ndarray] = None
        self._cached_K: Optional[np.ndarray] = None
        self._cached_u_d: Optional[np.ndarray] = None

    def reset(self, x: float, y: float, theta: float):
        self.state = np.zeros(10)
        self.world_pose = np.array([x, y, theta], dtype=float)
        self.state[self.IDX_PHI] = float(theta)

    def _update_cache(self, l_l: float, l_r: float):
        if self._cached_l_l != l_l or self._cached_l_r != l_r:
            self._cached_l_l = l_l
            self._cached_l_r = l_r
            self._cached_A = self._compute_A_matrix(l_l, l_r)
            self._cached_B = self._compute_B_matrix(l_l, l_r)
            self._cached_K = PolynomialFit.get_K_matrix(l_l, l_r)
            self._cached_u_d = PolynomialFit.get_u_d(l_l, l_r)

    def _compute_A_matrix(self, l_l: float, l_r: float) -> np.ndarray:
        """计算状态空间A矩阵"""
        A = np.zeros((10, 10))
        
        # 运动学关系 (速度到位置)
        A[0, 1] = 1  # ds/dt = ds
        A[2, 3] = 1  # dphi/dt = dphi  
        A[4, 5] = 1  # dtheta_l_l/dt = dtheta_l_l
        A[6, 7] = 1  # dtheta_l_r/dt = dtheta_l_r
        A[8, 9] = 1  # dtheta_b/dt = dtheta_b
        
        # A矩阵多项式系数
        A[1, 4] = PolynomialFit.eval_poly33(
            np.array([-14.988696, 0.016197, -1.661738, -0.006847, -0.006138,
                0.60926, 0.001655, 0.001976, 0.002136, -0.144306]), l_l, l_r)
        A[1, 6] = PolynomialFit.eval_poly33(
            np.array([-14.988696, -1.661738, 0.016197, 0.60926, -0.006138,
                -0.006847, -0.144306, 0.002136, 0.001976, 0.001655]), l_l, l_r)
        A[3, 4] = PolynomialFit.eval_poly33(
            np.array([-21.326009, -0.023151, -2.378261, 0.009805, 0.008931,
                0.874911, -0.002364, -0.002866, -0.003141, -0.20748]), l_l, l_r)
        A[3, 6] = PolynomialFit.eval_poly33(
            np.array([21.326009, 2.378261, 0.023151, -0.874911, -0.008931,
                -0.009805, 0.20748, 0.003141, 0.002866, 0.002364]), l_l, l_r)
        A[5, 4] = PolynomialFit.eval_poly33(
            np.array([177.814842, -0.002676, -33.716541, 0.000862, 0.001306,
                1.506241, -0.000102, -0.000351, -0.000352, 1.536279]), l_l, l_r)
        A[5, 6] = PolynomialFit.eval_poly33(
            np.array([-1.401778, 0.375475, 0.232589, -0.154863, -0.056214,
                0.032082, 0.042702, 0.019666, -0.009315, -0.032746]), l_l, l_r)
        A[7, 4] = PolynomialFit.eval_poly33(
            np.array([-1.401778, 0.232589, 0.375475, 0.032082, -0.056214,
                -0.154863, -0.032746, -0.009315, 0.019666, 0.042702]), l_l, l_r)
        A[7, 6] = PolynomialFit.eval_poly33(
            np.array([177.814842, -33.716541, -0.002676, 1.506241, 0.001306,
                0.000862, 1.536279, -0.000352, -0.000351, -0.000102]), l_l, l_r)
        A[9, 4] = PolynomialFit.eval_poly33(
            np.array([-2.580536, 0.003393, -0.362581, -0.001394, -0.001271,
                0.135965, 0.000322, 0.000402, 0.000442, -0.033092]), l_l, l_r)
        A[9, 6] = PolynomialFit.eval_poly33(
            np.array([-2.580536, -0.362581, 0.003393, 0.135965, -0.001271,
                -0.001394, -0.033092, 0.000442, 0.000402, 0.000322]), l_l, l_r)
        A[9, 8] = PolynomialFit.eval_poly33(
            np.array([5.77224, -0.0, -0.0, -0.0, -0.0,
                -0.0, 0.0, 0.0, 0.0, 0.0]), l_l, l_r)
        
        return A
    
    def _compute_B_matrix(self, l_l: float, l_r: float) -> np.ndarray:
        """计算状态空间B矩阵"""
        B = np.zeros((10, 4))
        
        B[1, 0] = PolynomialFit.eval_poly33(
            np.array([3.077659, 0.017621, 0.06205, -0.006367, 0.001413,
                -0.063237, 0.001538, -0.000452, -0.000576, 0.022175]), l_l, l_r)
        B[1, 1] = PolynomialFit.eval_poly33(
            np.array([3.077659, 0.06205, 0.017621, -0.063237, 0.001413,
                -0.006367, 0.022175, -0.000576, -0.000452, 0.001538]), l_l, l_r)
        B[1, 2] = PolynomialFit.eval_poly33(
            np.array([-0.647006, 0.000644, 0.120827, -0.000357, -0.000662,
                -0.003677, 8.6e-05, 0.000213, 0.000312, -0.006363]), l_l, l_r)
        B[1, 3] = PolynomialFit.eval_poly33(
            np.array([-0.647006, 0.120827, 0.000644, -0.003677, -0.000662,
                -0.000357, -0.006363, 0.000312, 0.000213, 8.6e-05]), l_l, l_r)
        B[3, 0] = PolynomialFit.eval_poly33(
            np.array([-1.037352, -0.025198, 0.090258, 0.009105, -0.001912,
                -0.091836, -0.002196, 0.000617, 0.000797, 0.032195]), l_l, l_r)
        B[3, 1] = PolynomialFit.eval_poly33(
            np.array([1.037352, -0.090258, 0.025198, 0.091836, 0.001912,
                -0.009105, -0.032195, -0.000797, -0.000617, 0.002196]), l_l, l_r)
        B[3, 2] = PolynomialFit.eval_poly33(
            np.array([-0.920826, -0.000919, 0.171499, 0.000512, 0.000957,
                -0.004296, -0.000123, -0.000307, -0.000454, -0.009443]), l_l, l_r)
        B[3, 3] = PolynomialFit.eval_poly33(
            np.array([0.920826, -0.171499, 0.000919, 0.004296, -0.000957,
                -0.000512, 0.009443, 0.000454, 0.000307, 0.000123]), l_l, l_r)
        B[5, 0] = PolynomialFit.eval_poly33(
            np.array([-27.603564, -0.002862, 7.515111, 0.000768, 0.000163,
                -1.606377, -9.1e-05, -4.4e-05, 0.000166, 0.191701]), l_l, l_r)
        B[5, 1] = PolynomialFit.eval_poly33(
            np.array([-1.486576, -0.075268, 0.239488, 0.040502, 0.012474,
                0.029811, -0.014001, -0.005141, 0.002071, -0.030427]), l_l, l_r)
        B[5, 2] = PolynomialFit.eval_poly33(
            np.array([7.58546, -0.00011, -3.672108, 4.6e-05, 0.000119,
                1.386931, -5e-06, -3.2e-05, -5.4e-05, -0.343371]), l_l, l_r)
        B[5, 3] = PolynomialFit.eval_poly33(
            np.array([-0.05873, 0.032672, 0.010305, -0.022493, -0.006044,
                0.001673, 0.00889, 0.002856, -0.001002, -0.001708]), l_l, l_r)
        B[7, 0] = PolynomialFit.eval_poly33(
            np.array([-1.486576, 0.239488, -0.075268, 0.029811, 0.012474,
                0.040502, -0.030427, 0.002071, -0.005141, -0.014001]), l_l, l_r)
        B[7, 1] = PolynomialFit.eval_poly33(
            np.array([-27.603564, 7.515111, -0.002862, -1.606377, 0.000163,
                0.000768, 0.191701, 0.000166, -4.4e-05, -9.1e-05]), l_l, l_r)
        B[7, 2] = PolynomialFit.eval_poly33(
            np.array([-0.05873, 0.010305, 0.032672, 0.001673, -0.006044,
                -0.022493, -0.001708, -0.001002, 0.002856, 0.00889]), l_l, l_r)
        B[7, 3] = PolynomialFit.eval_poly33(
            np.array([7.58546, -3.672108, -0.00011, 1.386931, 0.000119,
                4.6e-05, -0.343371, -5.4e-05, -3.2e-05, -5e-06]), l_l, l_r)
        B[9, 0] = PolynomialFit.eval_poly33(
            np.array([0.074344, 0.003685, 0.022329, -0.001296, 0.000293,
                -0.017717, 0.000299, -9.2e-05, -0.000119, 0.006183]), l_l, l_r)
        B[9, 1] = PolynomialFit.eval_poly33(
            np.array([0.074344, 0.022329, 0.003685, -0.017717, 0.000293,
                -0.001296, 0.006183, -0.000119, -9.2e-05, 0.000299]), l_l, l_r)
        B[9, 2] = PolynomialFit.eval_poly33(
            np.array([-2.317397, 0.000135, 0.017755, -7.3e-05, -0.000137,
                0.002636, 1.7e-05, 4.3e-05, 6.5e-05, -0.002493]), l_l, l_r)
        B[9, 3] = PolynomialFit.eval_poly33(
            np.array([-2.317397, 0.017755, 0.000135, 0.002636, -0.000137,
                -7.3e-05, -0.002493, 6.5e-05, 4.3e-05, 1.7e-05]), l_l, l_r)
        
        return B

    def step_lqr(self, s_ref: float, v_ref: float, theta_ref: float, omega_ref: float, dt: float, u_disturb: np.ndarray) -> None:
        """1000Hz 子步。

        - 使用参考状态 x_ref = [s_ref, v_ref, theta_ref, omega_ref, 0,0,0,0,0,0]
        - 计算 u = -K*(x-x_ref)
        - 输入侧叠加扰动 u_disturb（偏置+噪声）
        """
        l_l = float(np.clip(self.params.l_l, self.params.h_min, self.params.h_max))
        l_r = float(np.clip(self.params.l_r, self.params.h_min, self.params.h_max))
        self._update_cache(l_l, l_r)

        assert self._cached_A is not None
        assert self._cached_B is not None
        assert self._cached_K is not None
        assert self._cached_u_d is not None

        x_ref = np.zeros(10)
        x_ref[self.IDX_S] = float(s_ref)
        x_ref[self.IDX_DS] = float(v_ref)
        x_ref[self.IDX_PHI] = float(theta_ref)
        x_ref[self.IDX_DPHI] = float(omega_ref)

        x_err = self.state - x_ref
        x_err[self.IDX_PHI] = angle_diff(float(self.state[self.IDX_PHI]), float(theta_ref))

        u = -self._cached_K @ x_err + self._cached_u_d

        # 扰动叠加 + 限幅
        u = u + u_disturb
        u[0] = float(np.clip(u[0], -self.params.T_w_max, self.params.T_w_max))
        u[1] = float(np.clip(u[1], -self.params.T_w_max, self.params.T_w_max))
        u[2] = float(np.clip(u[2], -self.params.T_b_max, self.params.T_b_max))
        u[3] = float(np.clip(u[3], -self.params.T_b_max, self.params.T_b_max))

        x_dot = self._cached_A @ self.state + self._cached_B @ u
        self.state = self.state + x_dot * dt

        # 世界位姿更新
        v = float(self.state[self.IDX_DS])
        yaw = float(self.state[self.IDX_PHI])
        self.world_pose[0] += v * math.cos(float(self.world_pose[2])) * dt
        self.world_pose[1] += v * math.sin(float(self.world_pose[2])) * dt
        self.world_pose[2] = yaw

    @property
    def x(self) -> float:
        return float(self.world_pose[0])

    @property
    def y(self) -> float:
        return float(self.world_pose[1])

    @property
    def s(self) -> float:
        return float(self.state[self.IDX_S])

    @property
    def theta(self) -> float:
        return float(self.world_pose[2])

    @property
    def v(self) -> float:
        return float(self.state[self.IDX_DS])

    @property
    def omega(self) -> float:
        return float(self.state[self.IDX_DPHI])


# =============================================================================
# nav map helpers (参考 path_follower/path_planner/nav_map.cpp)
# =============================================================================


class CostMap2D:
    def __init__(self, msg: OccupancyGrid):
        self.width = int(msg.info.width)
        self.height = int(msg.info.height)
        self.resolution = float(msg.info.resolution)
        self.origin_x = float(msg.info.origin.position.x)
        self.origin_y = float(msg.info.origin.position.y)

        # OccupancyGrid.data 是 int8；工程里实际把它当作 uint8 cost(0~255) 来用
        raw = np.asarray(msg.data, dtype=np.int8)
        self.data = raw.astype(np.uint8, copy=False)
        if self.data.size != self.width * self.height:
            raise ValueError(f"CostMap data size mismatch: got {self.data.size}, expected {self.width*self.height}")

    def map_coord_to_grid(self, x: float, y: float) -> tuple[float, float]:
        return ((x - self.origin_x) / self.resolution, (y - self.origin_y) / self.resolution)

    def is_valid_grid_coord_i(self, xi: int, yi: int) -> bool:
        return 0 <= xi < self.width and 0 <= yi < self.height

    def is_valid_grid_coord_d(self, gx: float, gy: float) -> bool:
        return 0.0 <= gx and gx + 1.0 <= float(self.width) and 0.0 <= gy and gy + 1.0 <= float(self.height)

    def at(self, xi: int, yi: int) -> int:
        if self.is_valid_grid_coord_i(xi, yi):
            return int(self.data[yi * self.width + xi])
        return 255

    def interpolate_grid(self, gx: float, gy: float) -> float:
        if not self.is_valid_grid_coord_d(gx, gy):
            return 255.0

        x0 = int(gx)
        y0 = int(gy)
        x1 = x0 + 1
        y1 = y0 + 1
        dx = float(gx - x0)
        dy = float(gy - y0)

        c00 = self.at(x0, y0)
        c10 = self.at(x1, y0)
        c01 = self.at(x0, y1)
        c11 = self.at(x1, y1)

        return (1 - dx) * (1 - dy) * c00 + dx * (1 - dy) * c10 + (1 - dx) * dy * c01 + dx * dy * c11

    def cost_at_map(self, x: float, y: float) -> Optional[float]:
        gx, gy = self.map_coord_to_grid(x, y)
        if not self.is_valid_grid_coord_d(gx, gy):
            return None
        return float(self.interpolate_grid(gx, gy))

    def gradient_grid(self, gx: float, gy: float) -> np.ndarray:
        samples = 2
        x = int(gx)
        y = int(gy)

        sum_gx = 0.0
        sum_gy = 0.0
        for i in range(1, samples + 1):
            sum_gx += (self.at(x + i, y) - self.at(x - i, y)) / (i * 2.0)
            sum_gy += (self.at(x, y + i) - self.at(x, y - i)) / (i * 2.0)
        return np.array([sum_gx / samples, sum_gy / samples], dtype=float)

    def gradient_at_map(self, x: float, y: float) -> Optional[np.ndarray]:
        gx, gy = self.map_coord_to_grid(x, y)
        if not self.is_valid_grid_coord_d(gx, gy):
            return None
        return self.gradient_grid(gx, gy)


class DirectionMap2D:
    def __init__(self, msg: Image, resolution: float, origin_x: float, origin_y: float):
        self.width = int(msg.width)
        self.height = int(msg.height)
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)

        raw = np.frombuffer(msg.data, dtype=np.uint8)

        # map_server 现在发布 8UC3：B=dx, G=dy, R=step_mode
        # 兼容旧版 8UC2
        if msg.encoding in ("8UC3", "8UC3; compressed"):
            nch = 3
        elif msg.encoding in ("8UC2", "8UC2; compressed"):
            nch = 2
        else:
            total_pixels = int(msg.width) * int(msg.height)
            if total_pixels > 0 and raw.size % total_pixels == 0:
                nch = raw.size // total_pixels
            else:
                raise ValueError(f"Unsupported direction map encoding: {msg.encoding}")

        if msg.step <= 0:
            raise ValueError("DirectionMap image step is invalid")
        if raw.size < int(msg.step) * int(msg.height):
            raise ValueError("DirectionMap image data too short")

        row_bytes = int(msg.step)
        pixels_per_row = int(msg.width) * nch
        mat = raw.reshape((int(msg.height), row_bytes))[:, :pixels_per_row]
        pix = mat.reshape((int(msg.height), int(msg.width), nch))

        # direction is always in the first 2 channels
        p0 = pix[:, :, 0]
        p1 = pix[:, :, 1]
        mask_zero = ((p0 == 0) & (p1 == 0)) | ((p0 == 128) & (p1 == 128))

        vec = pix[:, :, :2].astype(np.float32)
        vec = (vec - 128.0) / 128.0
        vec[mask_zero, :] = 0.0
        self.data = vec  # (H,W,2)

    def map_coord_to_grid(self, x: float, y: float) -> tuple[float, float]:
        return ((x - self.origin_x) / self.resolution, (y - self.origin_y) / self.resolution)

    def is_valid_grid_coord_d(self, gx: float, gy: float) -> bool:
        return 0.0 <= gx and gx + 1.0 <= float(self.width) and 0.0 <= gy and gy + 1.0 <= float(self.height)

    def at(self, xi: int, yi: int) -> np.ndarray:
        if 0 <= xi < self.width and 0 <= yi < self.height:
            return self.data[yi, xi, :].astype(float, copy=False)
        return np.array([0.0, 0.0], dtype=float)

    def interpolate_grid(self, gx: float, gy: float) -> np.ndarray:
        if not self.is_valid_grid_coord_d(gx, gy):
            return np.array([0.0, 0.0], dtype=float)

        x0 = int(gx)
        y0 = int(gy)
        x1 = x0 + 1
        y1 = y0 + 1
        dx = float(gx - x0)
        dy = float(gy - y0)

        v00 = self.at(x0, y0)
        v10 = self.at(x1, y0)
        v01 = self.at(x0, y1)
        v11 = self.at(x1, y1)

        return (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 + (1 - dx) * dy * v01 + dx * dy * v11

    def direction_at_map(self, x: float, y: float) -> Optional[np.ndarray]:
        gx, gy = self.map_coord_to_grid(x, y)
        if not self.is_valid_grid_coord_d(gx, gy):
            return None
        return self.interpolate_grid(gx, gy)


# =============================================================================
# 动态障碍物运动状态
# =============================================================================


class ObstacleState:
    """动态障碍物运动状态机。

    两种轨迹类型：
      circle —— 以随机中心做圆周运动，顺/逆时针随机选取。
      line   —— 在随机方向的线段上来回运动；临近端点时自动制动，反向后重新
                加速，全程通过加速度限幅保证速度连续（不会突变）。

    两种速度类型（均受 max_accel 限制过渡）：
      constant    —— 恒定 max_speed。
      oscillating —— 速度以 oscillate_freq 频率在 0~max_speed 正弦振荡。
    """

    def __init__(
        self,
        spec: ObstacleSpec,
        rng: np.random.Generator,
        x_min: float, x_max: float,
        y_min: float, y_max: float,
    ) -> None:
        self.spec = spec
        self._sim_time: float = 0.0
        self._current_speed: float = 0.0  # 当前实际速度（标量，始终 >= 0）

        if spec.motion_type == "circle":
            r = spec.circle_radius
            # 圆心范围：确保整条轨道都在 spawn 范围内
            cx_lo = x_min + r;  cx_hi = x_max - r
            cy_lo = y_min + r;  cy_hi = y_max - r
            if cx_lo > cx_hi:
                cx_lo = cx_hi = (x_min + x_max) * 0.5
            if cy_lo > cy_hi:
                cy_lo = cy_hi = (y_min + y_max) * 0.5
            self._center = np.array([
                rng.uniform(cx_lo, cx_hi),
                rng.uniform(cy_lo, cy_hi),
            ], dtype=float)
            self._angle: float = float(rng.uniform(0.0, 2.0 * math.pi))
            # 随机顺/逆时针
            self._orbit_sign: float = float(rng.choice([-1.0, 1.0]))
            # 直线用字段（circle 模式下置零，避免 Optional）
            self._line_start = np.zeros(2, dtype=float)
            self._line_dir = np.zeros(2, dtype=float)
            self._progress: float = 0.0
            self._going_forward: bool = True
        else:  # "line"
            ll = spec.line_length
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            dir_x = math.cos(angle)
            dir_y = math.sin(angle)
            # 约束起点使整段线段 [start, start + dir*ll] 在 spawn 范围内
            sx_lo = x_min - min(0.0, dir_x * ll);  sx_hi = x_max - max(0.0, dir_x * ll)
            sy_lo = y_min - min(0.0, dir_y * ll);  sy_hi = y_max - max(0.0, dir_y * ll)
            if sx_lo > sx_hi:
                sx_lo = sx_hi = (x_min + x_max) * 0.5
            if sy_lo > sy_hi:
                sy_lo = sy_hi = (y_min + y_max) * 0.5
            self._line_start = np.array([
                rng.uniform(sx_lo, sx_hi),
                rng.uniform(sy_lo, sy_hi),
            ], dtype=float)
            self._line_dir = np.array([dir_x, dir_y], dtype=float)
            self._progress = 0.0
            self._going_forward = True
            # circle 用字段置零
            self._center = np.zeros(2, dtype=float)
            self._angle = 0.0
            self._orbit_sign = 1.0

    # ------------------------------------------------------------------
    # 期望速度（不含加速度限制）
    # ------------------------------------------------------------------

    def _desired_speed(self) -> float:
        """根据速度类型和轨迹位置计算期望标量速度（>= 0）。"""
        spec = self.spec

        # 基础速度
        if spec.speed_type == "oscillating":
            phase = 2.0 * math.pi * spec.oscillate_freq * self._sim_time
            base = spec.max_speed * (0.5 + 0.5 * math.sin(phase))
        else:
            base = spec.max_speed

        # 直线轨迹：临近端点时进行运动学制动，保证能在端点平稳停下
        if spec.motion_type == "line":
            dist_to_end = (
                spec.line_length - self._progress
                if self._going_forward
                else self._progress
            )
            # 当前速度对应的制动距离 d = v² / (2a)
            brake_dist = self._current_speed ** 2 / (2.0 * spec.max_accel + 1e-9)
            if dist_to_end < brake_dist:
                # 计算此距离下的最大允许速度，防止冲出端点
                safe = math.sqrt(max(0.0, 2.0 * spec.max_accel * dist_to_end))
                base = min(base, safe)

        return max(0.0, float(base))

    # ------------------------------------------------------------------
    # 状态推进
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        """推进障碍物状态 dt 秒（dt 应与 LQR 步长一致以保持精度）。"""
        self._sim_time += dt
        spec = self.spec

        # 用加速度限幅平滑过渡到期望速度
        desired = self._desired_speed()
        delta = desired - self._current_speed
        max_delta = spec.max_accel * dt
        self._current_speed += float(np.clip(delta, -max_delta, max_delta))
        self._current_speed = max(0.0, self._current_speed)

        if spec.motion_type == "circle":
            omega = self._orbit_sign * self._current_speed / (spec.circle_radius + 1e-9)
            self._angle += omega * dt
        else:  # "line"
            if self._going_forward:
                self._progress += self._current_speed * dt
                if self._progress >= spec.line_length:
                    self._progress = float(spec.line_length)
                    self._going_forward = False  # 到达端点，反向
            else:
                self._progress -= self._current_speed * dt
                if self._progress <= 0.0:
                    self._progress = 0.0
                    self._going_forward = True   # 回到起点，正向

    # ------------------------------------------------------------------
    # 当前位置
    # ------------------------------------------------------------------

    @property
    def pos(self) -> tuple[float, float]:
        """返回当前世界坐标 (x, y)。"""
        spec = self.spec
        if spec.motion_type == "circle":
            x = self._center[0] + spec.circle_radius * math.cos(self._angle)
            y = self._center[1] + spec.circle_radius * math.sin(self._angle)
        else:
            x = self._line_start[0] + self._line_dir[0] * self._progress
            y = self._line_start[1] + self._line_dir[1] * self._progress
        return float(x), float(y)


# =============================================================================
# ROS2 Node
# =============================================================================


@dataclass
class OdomSample:
    stamp: rclpy.time.Time
    x: float
    y: float
    theta: float
    v: float
    w: float


class WheelLegLqrFollowSimNode(Node):
    def __init__(self):
        super().__init__("wheel_leg_lqr_follow_sim")

        self.cfg = SimConfig()
        self.params = RobotParams()

        self._dt_lqr = 1.0 / self.cfg.LQR_FREQ_HZ
        self._cmd_timeout = Duration(seconds=float(self.cfg.CMD_TIMEOUT_SEC))
        self._odom_delay = Duration(seconds=float(self.cfg.ODOM_DELAY_SEC))

        self._rng = np.random.default_rng()

        # plant
        self.dyn = WheelLegDynamics(self.params)
        self.dyn.reset(self.cfg.INIT_X, self.cfg.INIT_Y, self.cfg.INIT_THETA)

        # energy simulation
        self._energy = float(self.cfg.CAPACITOR_INIT_ENERGY)
        self._current_power = 0.0
        _lp_dt = 1.0 / self.cfg.LQR_FREQ_HZ
        _lp_wc = 2.0 * math.pi * self.cfg.POWER_LOWPASS_HZ
        self._pwr_lp_alpha = _lp_dt * _lp_wc / (1.0 + _lp_dt * _lp_wc)
        self._v_filt = 0.0
        self._w_filt = 0.0
        self._v_filt_prev = 0.0
        self._w_filt_prev = 0.0

        # command targets (MPC-rate input -> 1000Hz rate-limited application)
        self._last_cmd_time = self.get_clock().now()
        self._spin_slow = False
        self._spin_fast = False

        self._v_target = 0.0
        self._w_target = 0.0
        self._theta_target = float(self.cfg.INIT_THETA)

        self._v_applied = 0.0
        self._w_applied = 0.0
        self._w_prev = 0.0

        # environment interaction state
        self._step_stuck = False

        self._global_cost_map: Optional[CostMap2D] = None
        self._local_cost_map: Optional[CostMap2D] = None
        self._direction_map: Optional[DirectionMap2D] = None

        # s reference
        self._s_ref = float(self.dyn.s)
        self._was_spin_mode = False

        # odom delay buffer
        buffer_sec = max(1.0, float(self.cfg.ODOM_DELAY_SEC) + 1.0)
        self._odom_buf: deque[OdomSample] = deque(maxlen=int(buffer_sec * self.cfg.LQR_FREQ_HZ) + 50)

        # 动态障碍物状态列表
        self._obstacle_states: list[ObstacleState] = [
            ObstacleState(
                spec, self._rng,
                self.cfg.OBSTACLE_SPAWN_X_MIN, self.cfg.OBSTACLE_SPAWN_X_MAX,
                self.cfg.OBSTACLE_SPAWN_Y_MIN, self.cfg.OBSTACLE_SPAWN_Y_MAX,
            )
            for spec in self.cfg.OBSTACLE_SPECS
        ]
        self._unit_sphere_points = self._build_unit_sphere_samples()

        # pubs
        self.joint_state_pub = self.create_publisher(JointState, self.cfg.TOPIC_JOINT, 2)
        self.odom_pub = self.create_publisher(Odometry, self.cfg.TOPIC_ODOM, 2)
        self.imu_pub = self.create_publisher(Imu, self.cfg.TOPIC_IMU, 2)
        self.status_pub = self.create_publisher(ChassisStatus, self.cfg.TOPIC_STATUS, 2)
        self.cloud_pub = self.create_publisher(PointCloud2, self.cfg.TOPIC_CLOUD, 2)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # sub
        self.create_subscription(ChassisCmd, self.cfg.TOPIC_CMD, self._cmd_cb, 2)

        # map subs
        self.create_subscription(OccupancyGrid, self.cfg.TOPIC_GLOBAL_COST_MAP, self._on_global_cost_map, 1)
        self.create_subscription(CostMaps, self.cfg.TOPIC_LOCAL_COST_MAPS, self._on_local_cost_maps, 1)
        self.create_subscription(Image, self.cfg.TOPIC_GLOBAL_DIRECTION_MAP, self._on_global_direction_map, 1)

        # timers
        self._lqr_timer = self.create_timer(self._dt_lqr, self._on_lqr)
        self._imu_timer = self.create_timer(1.0 / self.cfg.IMU_PUB_HZ, self._pub_imu)
        self._joint_timer = self.create_timer(1.0 / self.cfg.JOINT_PUB_HZ, self._pub_joint)
        self._status_timer = self.create_timer(1.0 / self.cfg.STATUS_PUB_HZ, self._pub_status)
        self._odom_timer = self.create_timer(1.0 / self.cfg.ODOM_PUB_HZ, self._pub_odom)
        self._cloud_timer = self.create_timer(1.0 / self.cfg.CLOUD_PUB_HZ, self._pub_cloud)
        self._tf_timer = self.create_timer(1.0 / self.cfg.TF_PUB_HZ, self._pub_tf)

        self.get_logger().info(
            "WheelLegLqrFollowSimNode started\n"
            f"  LQR: {self.cfg.LQR_FREQ_HZ} Hz\n"
            f"  IMU: {self.cfg.IMU_PUB_HZ} Hz\n"
            f"  Joint: {self.cfg.JOINT_PUB_HZ} Hz\n"
            f"  Status: {self.cfg.STATUS_PUB_HZ} Hz\n"
            f"  Odom: {self.cfg.ODOM_PUB_HZ} Hz (delay {self.cfg.ODOM_DELAY_SEC}s)\n"
            f"  Cloud: {self.cfg.CLOUD_PUB_HZ} Hz\n"
            f"  TF(odom->map): {self.cfg.TF_PUB_HZ} Hz"
        )

    def _build_unit_sphere_samples(self) -> list[tuple[float, float, float]]:
        """生成单位球面均匀采样点（Fibonacci sphere）。"""
        r = max(1e-6, float(self.cfg.OBSTACLE_CLOUD_RADIUS_M))
        density = max(0.0, float(self.cfg.OBSTACLE_CLOUD_DENSITY_PTS_PER_M2))
        area = 4.0 * math.pi * r * r
        est_points = int(area * density)
        n = max(int(self.cfg.OBSTACLE_CLOUD_MIN_POINTS), est_points)

        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        inv_n = 1.0 / float(n)
        points: list[tuple[float, float, float]] = []

        for i in range(n):
            z = 1.0 - 2.0 * ((i + 0.5) * inv_n)
            rr = math.sqrt(max(0.0, 1.0 - z * z))
            a = golden_angle * i
            points.append((rr * math.cos(a), rr * math.sin(a), z))

        return points

    # ------------------------
    # map callbacks
    # ------------------------

    def _on_global_cost_map(self, msg: OccupancyGrid) -> None:
        try:
            self._global_cost_map = CostMap2D(msg)
        except Exception as ex:
            self.get_logger().error(f"Failed to parse global cost map: {ex}")
            self._global_cost_map = None

    def _on_local_cost_maps(self, msg: CostMaps) -> None:
        if not msg.maps:
            self._local_cost_map = None
            return
        try:
            self._local_cost_map = CostMap2D(msg.maps[0])
        except Exception as ex:
            self.get_logger().error(f"Failed to parse cost maps: {ex}")
            self._local_cost_map = None

    def _on_global_direction_map(self, msg: Image) -> None:
        if self._global_cost_map is None:
            self.get_logger().warn("Received direction map but global cost map is not ready yet")
            return
        try:
            dm = DirectionMap2D(
                msg,
                resolution=self._global_cost_map.resolution,
                origin_x=self._global_cost_map.origin_x,
                origin_y=self._global_cost_map.origin_y,
            )
            if dm.width != self._global_cost_map.width or dm.height != self._global_cost_map.height:
                raise ValueError(
                    f"Direction map size ({dm.width},{dm.height}) does not match cost map ({self._global_cost_map.width},{self._global_cost_map.height})"
                )
            self._direction_map = dm
        except Exception as ex:
            self.get_logger().error(f"Failed to parse direction map: {ex}")
            self._direction_map = None

    # ------------------------
    # environment helpers
    # ------------------------

    def _cost_at(self, x: float, y: float) -> Optional[float]:
        costs: list[float] = []
        if self._global_cost_map is not None:
            c = self._global_cost_map.cost_at_map(x, y)
            if c is not None:
                costs.append(float(c))
        if self._local_cost_map is not None:
            c = self._local_cost_map.cost_at_map(x, y)
            if c is not None:
                costs.append(float(c))
        if not costs:
            return None
        return float(max(costs))

    def _gradient_at(self, x: float, y: float) -> Optional[np.ndarray]:
        # 取当前位置 cost 更大的那张图的梯度，避免局部/全局不一致时方向跳变
        candidates: list[tuple[float, np.ndarray]] = []
        if self._global_cost_map is not None:
            c = self._global_cost_map.cost_at_map(x, y)
            g = self._global_cost_map.gradient_at_map(x, y)
            if c is not None and g is not None:
                candidates.append((float(c), g))
        if self._local_cost_map is not None:
            c = self._local_cost_map.cost_at_map(x, y)
            g = self._local_cost_map.gradient_at_map(x, y)
            if c is not None and g is not None:
                candidates.append((float(c), g))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]

    def _find_escape_dir(self, x: float, y: float, step: float) -> Optional[np.ndarray]:
        base_cost = self._cost_at(x, y)
        if base_cost is None:
            return None
        best_cost = float(base_cost)
        best_dir: Optional[np.ndarray] = None
        for k in range(16):
            ang = 2.0 * math.pi * (k / 16.0)
            dx = math.cos(ang) * step
            dy = math.sin(ang) * step
            c = self._cost_at(x + dx, y + dy)
            if c is None:
                continue
            if float(c) < best_cost:
                best_cost = float(c)
                best_dir = np.array([dx, dy], dtype=float)
        if best_dir is None:
            return None
        n = float(np.linalg.norm(best_dir))
        if n <= 1e-9:
            return None
        return best_dir / n

    def _collision_flags(self, x: float, y: float, yaw: float) -> tuple[bool, bool, Optional[float]]:
        r = float(self.cfg.ROBOT_INSCRIBED_RADIUS_M + self.cfg.COLLISION_LOOKAHEAD_M)
        fx = x + r * math.cos(yaw)
        fy = y + r * math.sin(yaw)
        rx = x - r * math.cos(yaw)
        ry = y - r * math.sin(yaw)

        c_front = self._cost_at(fx, fy)
        c_rear = self._cost_at(rx, ry)
        c_center = self._cost_at(x, y)

        thr = float(self.cfg.OBSTACLE_COST_THRESHOLD)
        front_hit = (c_front is not None) and (float(c_front) >= thr)
        rear_hit = (c_rear is not None) and (float(c_rear) >= thr)
        return front_hit, rear_hit, c_center

    def _step_info(self, x: float, y: float) -> tuple[float, float]:
        # returns: (step_norm, heading_err)
        # - step_norm: 台阶方向场模长
        # - heading_err: 与台阶方向夹角（前后等价，取更小者）
        if self._direction_map is None:
            return 0.0, 0.0
        vec = self._direction_map.direction_at_map(x, y)
        if vec is None:
            return 0.0, 0.0

        vx = float(vec[0])
        vy = float(vec[1])
        n = float(math.hypot(vx, vy))
        if n <= 1e-9:
            return n, 0.0

        step_yaw = math.atan2(vy, vx)
        yaw = wrap_to_pi(self.dyn.theta)
        e1 = abs(angle_diff(yaw, step_yaw))
        e2 = abs(angle_diff(yaw, wrap_to_pi(step_yaw + math.pi)))
        return n, float(min(e1, e2))

    # ------------------------
    # cmd processing
    # ------------------------

    def _cmd_cb(self, msg: ChassisCmd) -> None:
        now = self.get_clock().now()
        self._last_cmd_time = now

        slow_spin = (msg.mode == 1)
        fast_spin = (msg.mode == 2)
        self._spin_slow = slow_spin
        self._spin_fast = fast_spin

        if slow_spin or fast_spin:
            # ignore v/omega; spin handled in _on_lqr
            return

        v = float(msg.velocity)
        w = float(msg.omega)

        if not is_finite(v):
            v = 0.0
        if not is_finite(w):
            w = 0.0

        # 目标限幅（速度/角速度 + 乘积）；加速度在 1000Hz 中做 rate-limit
        v, w = self._clamp_v_w_product(v, w)

        self._v_target = v
        self._w_target = w

    def _clamp_v_w_product(self, v: float, w: float) -> tuple[float, float]:
        v = float(np.clip(v, -self.cfg.MAX_VELOCITY, self.cfg.MAX_VELOCITY))
        w = float(np.clip(w, -self.cfg.MAX_PALSTANCE, self.cfg.MAX_PALSTANCE))

        vw = abs(v * w)
        if vw > self.cfg.MAX_V_W_PRODUCT and vw > 1e-9:
            scale = self.cfg.MAX_V_W_PRODUCT / vw
            s = math.sqrt(scale)
            v *= s
            w *= s
        return v, w

    @staticmethod
    def _rate_limit(cur: float, target: float, max_rate: float, dt: float) -> float:
        if dt <= 0.0:
            return float(cur)
        delta = target - cur
        max_delta = max_rate * dt
        delta = float(np.clip(delta, -max_delta, max_delta))
        return float(cur + delta)

    # ------------------------
    # lqr loop
    # ------------------------

    def _on_lqr(self) -> None:
        now = self.get_clock().now()
        timed_out = (now - self._last_cmd_time) > self._cmd_timeout

        spin_mode = (self._spin_slow or self._spin_fast) and (not timed_out)

        if timed_out:
            # offline: stop v/w, hold theta; exit spin
            self._spin_slow = False
            self._spin_fast = False
            self._v_target = 0.0
            self._w_target = 0.0
            self._theta_target = float(self._theta_target)

        if spin_mode:
            # set spin target continuously
            spin_w = self.cfg.SPIN_FAST_OMEGA if self._spin_fast else self.cfg.SPIN_SLOW_OMEGA
            last_sign = 1.0 if self.dyn.omega >= 0.0 else -1.0
            self._v_target = 0.0
            self._w_target = float(spin_w * last_sign)

        # ----- environment constraints (台阶/障碍物) -----
        v_des = float(self._v_target)
        w_des = float(self._w_target)

        x0 = float(self.dyn.x)
        y0 = float(self.dyn.y)
        yaw0 = wrap_to_pi(float(self.dyn.theta))

        step_norm, step_heading_err = self._step_info(x0, y0)
        on_step = step_norm >= float(self.cfg.STEP_NORM_THRESHOLD)

        # step stuck state machine
        if self._step_stuck:
            if step_norm < float(self.cfg.STEP_RELEASE_NORM_THRESHOLD):
                self._step_stuck = False

        entered_stuck = False
        if (not self._step_stuck) and on_step and (v_des > 0.0):
            if (step_heading_err > float(self.cfg.STEP_STUCK_HEADING_ERR_RAD)) or (abs(float(self.dyn.v)) < float(self.cfg.STEP_STUCK_MIN_SPEED_MPS)):
                self._step_stuck = True
                entered_stuck = True

        if self._step_stuck:
            # 卡台阶：立即停住，之后只允许倒车，且不响应角速度
            spin_mode = False
            v_des = float(min(v_des, 0.0))
            w_des = 0.0
            if entered_stuck:
                self._v_applied = 0.0
                self._w_applied = 0.0
                self.dyn.state[self.dyn.IDX_DS] = 0.0
                self.dyn.state[self.dyn.IDX_DPHI] = 0.0

        front_hit, rear_hit, center_cost = self._collision_flags(x0, y0, yaw0)
        if front_hit and v_des > 0.0:
            v_des = 0.0
        if rear_hit and v_des < 0.0:
            v_des = 0.0

        # capture previous applied omega for trapezoidal integration
        w_prev = float(self._w_applied)

        # apply 1000Hz accel limits
        self._v_applied = self._rate_limit(self._v_applied, v_des, self.cfg.MAX_ACCEL, self._dt_lqr)
        self._w_applied = self._rate_limit(self._w_applied, w_des, self.cfg.MAX_ANG_ACCEL, self._dt_lqr)

        # enforce max speed / max omega / product after rate-limit
        self._v_applied, self._w_applied = self._clamp_v_w_product(self._v_applied, self._w_applied)

        # collision hard clamp after rate-limit (避免 1ms 内继续向障碍物“渗透”)
        if front_hit and self._v_applied > 0.0:
            self._v_applied = 0.0
        if rear_hit and self._v_applied < 0.0:
            self._v_applied = 0.0
        if self._step_stuck:
            self._w_applied = 0.0

        # dt 级预测：防止单步穿透障碍物（尤其是薄障碍/高速度时）
        if self._v_applied != 0.0:
            pred_x = x0 + float(self._v_applied) * math.cos(yaw0) * float(self._dt_lqr)
            pred_y = y0 + float(self._v_applied) * math.sin(yaw0) * float(self._dt_lqr)
            next_front, next_rear, next_center_cost = self._collision_flags(pred_x, pred_y, yaw0)
            thr = float(self.cfg.OBSTACLE_COST_THRESHOLD)
            center_block = (next_center_cost is not None) and (float(next_center_cost) >= thr)
            if self._v_applied > 0.0 and (next_front or center_block):
                self._v_applied = 0.0
            if self._v_applied < 0.0 and (next_rear or center_block):
                self._v_applied = 0.0

        # update theta reference by trapezoidal integration of applied omega
        self._theta_target = wrap_to_pi(self._theta_target + 0.5 * (w_prev + self._w_applied) * self._dt_lqr)

        # remember previous omega
        self._w_prev = float(self._w_applied)

        # update s reference
        # Use current LQR response state s as the baseline to avoid s_ref runaway
        # during slow initial posture adjustment / acceleration.
        if spin_mode:
            if not self._was_spin_mode:
                self._s_ref = float(self.dyn.s)
        else:
            self._s_ref = float(self.dyn.s) + float(self._v_applied) * self._dt_lqr
        self._was_spin_mode = spin_mode

        # torque disturbance
        bias = np.array(self.cfg.TORQUE_BIAS_NM, dtype=float)
        std = np.array(self.cfg.TORQUE_NOISE_STD_NM, dtype=float)
        noise = self._rng.normal(0.0, 1.0, size=4) * std
        u_disturb = bias + noise

        # step plant
        self.dyn.step_lqr(
            s_ref=self._s_ref,
            v_ref=self._v_applied,
            theta_ref=self._theta_target,
            omega_ref=self._w_applied,
            dt=self._dt_lqr,
            u_disturb=u_disturb,
        )

        # obstacle push-out (若当前位置代价过高，则沿“下降梯度”方向挤开)
        cost_now = self._cost_at(self.dyn.x, self.dyn.y)
        if cost_now is not None and float(cost_now) >= float(self.cfg.OBSTACLE_PUSH_COST_THRESHOLD):
            grad = self._gradient_at(self.dyn.x, self.dyn.y)
            push_dir: Optional[np.ndarray] = None
            if grad is not None:
                away = -grad
                n = float(np.linalg.norm(away))
                if n > 1e-6:
                    push_dir = away / n
            if push_dir is None:
                # 梯度过小/平台区域：尝试找一个更低 cost 的方向
                step = 0.0
                if self._global_cost_map is not None:
                    step = float(self._global_cost_map.resolution)
                elif self._local_cost_map is not None:
                    step = float(self._local_cost_map.resolution)
                step = max(0.05, step)
                push_dir = self._find_escape_dir(self.dyn.x, self.dyn.y, step)

            if push_dir is not None:
                disp = float(self.cfg.OBSTACLE_PUSH_SPEED_MPS) * float(self._dt_lqr)
                self.dyn.world_pose[0] += float(push_dir[0]) * disp
                self.dyn.world_pose[1] += float(push_dir[1]) * disp

        # spin drift (小陀螺模式位置缓慢漂移)
        if spin_mode and float(self.cfg.SPIN_DRIFT_SPEED_MPS) > 0.0:
            dx = float(self.cfg.SPIN_DRIFT_DIR_X)
            dy = float(self.cfg.SPIN_DRIFT_DIR_Y)
            n = math.hypot(dx, dy)
            if n > 1e-9:
                ux = dx / n
                uy = dy / n
                disp = float(self.cfg.SPIN_DRIFT_SPEED_MPS) * float(self._dt_lqr)
                self.dyn.world_pose[0] += ux * disp
                self.dyn.world_pose[1] += uy * disp

        # ─── Power model & energy tracking ───
        v_now = float(self.dyn.v)
        w_now = float(self.dyn.omega)
        self._v_filt += self._pwr_lp_alpha * (v_now - self._v_filt)
        self._w_filt += self._pwr_lp_alpha * (w_now - self._w_filt)
        a_filt = (self._v_filt - self._v_filt_prev) / self._dt_lqr
        alpha_filt = (self._w_filt - self._w_filt_prev) / self._dt_lqr
        self._current_power = predict_chassis_power(self._v_filt, self._w_filt, a_filt, alpha_filt)
        self._energy += (self.cfg.RFR_PWR_LIMIT - self._current_power) * self._dt_lqr
        self._energy = float(np.clip(self._energy, 0.0, self.cfg.CAPACITOR_MAX_ENERGY))
        self._v_filt_prev = self._v_filt
        self._w_filt_prev = self._w_filt

        # push odom sample for delay
        self._odom_buf.append(
            OdomSample(
                stamp=now,
                x=self.dyn.x,
                y=self.dyn.y,
                theta=self.dyn.theta,
                v=self.dyn.v,
                w=self.dyn.omega,
            )
        )

        # 推进所有动态障碍物（与 LQR 同频，保证运动时序一致）
        for obs in self._obstacle_states:
            obs.update(self._dt_lqr)

    # ------------------------
    # publishers
    # ------------------------

    def _pub_joint(self) -> None:
        now = self.get_clock().now()
        msg = JointState()
        msg.stamp = now.to_msg()
        msg.yaw_angle = 0.0
        msg.pitch_angle = 0.0
        self.joint_state_pub.publish(msg)

    def _pub_status(self) -> None:
        msg = ChassisStatus()
        msg.velocity = float(self.dyn.v)
        msg.omega = float(self.dyn.omega)
        msg.leg_h = float((self.params.l_l + self.params.l_r) / 2.0)
        msg.leg_psi = float(
            (self.dyn.state[self.dyn.IDX_THETA_L_L]
             + self.dyn.state[self.dyn.IDX_THETA_L_R]) / 2.0
        )
        msg.leg_mode = 4
        msg.remaining_energy_supercap = int(np.clip(self._energy, -32768, 32767))
        msg.remaining_energy_buffercap = 0
        msg.curr_chassis_pwr = int(np.clip(self._current_power, -32768, 32767))
        msg.rfr_pwr_limit = int(np.clip(self.cfg.RFR_PWR_LIMIT, -32768, 32767))
        self.status_pub.publish(msg)

    def _pub_imu(self) -> None:
        now = self.get_clock().now()
        yaw = wrap_to_pi(self.dyn.theta)
        if self.cfg.IMU_YAW_NOISE_STD_RAD > 0.0:
            yaw = wrap_to_pi(yaw + float(self._rng.normal(0.0, self.cfg.IMU_YAW_NOISE_STD_RAD)))

        msg = Imu()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.cfg.FRAME_IMU_WORLD
        msg.orientation = euler_to_quaternion(0.0, 0.0, yaw)
        self.imu_pub.publish(msg)

    def _pop_delayed_odom(self, now: rclpy.time.Time) -> Optional[OdomSample]:
        if not self._odom_buf:
            return None
        ready_time = now - self._odom_delay + Duration(seconds=1/self.cfg.ODOM_PUB_HZ)
        chosen: Optional[OdomSample] = None
        while self._odom_buf and self._odom_buf[0].stamp <= ready_time:
            chosen = self._odom_buf.popleft()
        return chosen

    def _pub_odom(self) -> None:
        now = self.get_clock().now()
        sample = self._pop_delayed_odom(now)
        if sample is None:
            return

        x = float(sample.x)
        y = float(sample.y)
        yaw = wrap_to_pi(float(sample.theta))
        v = float(sample.v)
        w = float(sample.w)

        if self.cfg.ODOM_X_NOISE_STD_M > 0.0:
            x += float(self._rng.normal(0.0, self.cfg.ODOM_X_NOISE_STD_M))
        if self.cfg.ODOM_Y_NOISE_STD_M > 0.0:
            y += float(self._rng.normal(0.0, self.cfg.ODOM_Y_NOISE_STD_M))
        if self.cfg.ODOM_YAW_NOISE_STD_RAD > 0.0:
            yaw = wrap_to_pi(yaw + float(self._rng.normal(0.0, self.cfg.ODOM_YAW_NOISE_STD_RAD)))
        if self.cfg.ODOM_V_NOISE_STD_MPS > 0.0:
            v += float(self._rng.normal(0.0, self.cfg.ODOM_V_NOISE_STD_MPS))
        if self.cfg.ODOM_W_NOISE_STD_RADPS > 0.0:
            w += float(self._rng.normal(0.0, self.cfg.ODOM_W_NOISE_STD_RADPS))

        msg = Odometry()
        msg.header.stamp = (now - self._odom_delay).to_msg()
        msg.header.frame_id = self.cfg.FRAME_ODOM
        msg.child_frame_id = self.cfg.FRAME_IMU_LINK
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation = euler_to_quaternion(0.0, 0.0, yaw)
        msg.twist.twist.linear.x = v
        msg.twist.twist.angular.z = w
        self.odom_pub.publish(msg)

    def _pub_cloud(self) -> None:
        now = self.get_clock().now()
        cfg = self.cfg

        radius = float(cfg.OBSTACLE_CLOUD_RADIUS_M)
        center_z = float(cfg.OBSTACLE_CLOUD_CENTER_Z_M)
        unit_sphere_points = self._unit_sphere_points

        # 为每个障碍物生成球形点云（球心 z 固定，仅 x/y 随轨迹运动）
        points: list[tuple[float, float, float]] = []
        for obs in self._obstacle_states:
            ox, oy = obs.pos
            for ux, uy, uz in unit_sphere_points:
                points.append((
                    ox + radius * ux,
                    oy + radius * uy,
                    center_z + radius * uz,
                ))

        msg = PointCloud2()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "odom"
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12          # 3 × float32
        msg.row_step = 12 * len(points)
        msg.is_dense = True

        data = bytearray()
        for px, py, pz in points:
            data += struct.pack('<fff', px, py, pz)
        msg.data = bytes(data)

        self.cloud_pub.publish(msg)

    def _pub_tf(self) -> None:
        now = self.get_clock().now()
        tf = TransformStamped()
        tf.header.stamp = now.to_msg()
        tf.header.frame_id = "map"
        tf.child_frame_id = self.cfg.FRAME_ODOM
        tf.transform.translation.x = 0.0
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.4
        tf.transform.rotation.w = 1.0
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = 0.0
        self.static_tf_broadcaster.sendTransform(tf)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WheelLegLqrFollowSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
