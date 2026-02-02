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
from interfaces.msg import ChassisCmd, ChassisStatus, JointState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2, PointField
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))

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
# 配置区（统一写在最上面，直接改即可）
# =============================================================================


@dataclass
class SimConfig:
    # --- 频率 ---
    LQR_FREQ_HZ: float = 1000.0
    IMU_PUB_HZ: float = 100.0
    JOINT_PUB_HZ: float = 20.0
    STATUS_PUB_HZ: float = 20.0
    ODOM_PUB_HZ: float = 10.0
    CLOUD_PUB_HZ: float = 10.0
    TF_PUB_HZ: float = 1.0

    # --- 超时 ---
    CMD_TIMEOUT_SEC: float = 0.3

    # --- 初始位姿（世界坐标） ---
    INIT_X: float = 2.0
    INIT_Y: float = 5.0
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
    ODOM_DELAY_SEC: float = 0.03

    # --- 传感器噪声（标准差）---
    IMU_YAW_NOISE_STD_RAD: float = 0.0

    ODOM_X_NOISE_STD_M: float = 0.0
    ODOM_Y_NOISE_STD_M: float = 0.0
    ODOM_YAW_NOISE_STD_RAD: float = 0.0
    ODOM_V_NOISE_STD_MPS: float = 0.0
    ODOM_W_NOISE_STD_RADPS: float = 0.0

    # --- 力矩扰动（每个通道）---
    # u = [T_w_l, T_w_r, T_b_l, T_b_r]
    TORQUE_BIAS_NM: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    TORQUE_NOISE_STD_NM: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    # --- 话题名 ---
    TOPIC_CMD: str = "/path_follower/chassis_cmd"
    TOPIC_JOINT: str = "/serial_bridge/joint_state"
    TOPIC_ODOM: str = "/small_glim/odometry"
    TOPIC_IMU: str = "/serial_bridge/imu"
    TOPIC_STATUS: str = "/serial_bridge/chassis_status"
    TOPIC_CLOUD: str = "/small_glim/registered_cloud"

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
            [-5.023000,-25.489884,26.473160,3.461858,-54.311257,-1.720216,220.481946,-396.570980,409.659553,-139.339081],
            [-5.805952,-13.585640,26.279902,-36.364715,-20.742078,-37.847277,212.069710,-277.099968,257.633136,-50.990234],
            [-4.104694,29.205246,-27.187177,70.122963,-211.710441,146.527221,-98.141650,58.143976,244.586374,-207.601520],
            [-1.460273,14.918158,-15.521422,38.338970,-110.274762,76.160402,-48.099609,32.539876,119.995621,-105.670945],
            [-4.670559,3.289463,-38.750458,-66.912126,-94.578161,124.943879,222.653007,-148.599282,218.791507,-205.848041],
            [-1.413626,0.503095,-0.793935,-5.203305,-11.622457,-15.020677,26.805748,4.674458,-5.948188,14.749590],
            [-3.476153,-69.132520,44.817887,-431.135011,588.614052,-258.114023,612.388562,-326.257136,-458.331538,318.929699],
            [-0.938071,-6.393517,9.376492,-111.742335,135.153393,-59.204559,84.346325,31.143327,-170.429300,89.324244],
            [-7.504694,-33.887035,43.559019,74.405414,20.719287,-95.408413,2.811991,-179.934598,131.906037,41.627062],
            [-1.418912,-5.176475,8.722835,7.298926,6.056789,-21.722833,9.282879,-34.395190,21.574021,13.692190],
        ],
        [
            [-7.146937,65.633922,-41.173444,27.568227,-305.276850,217.779767,-112.564984,115.529419,361.729938,-308.207287],
            [-7.597154,56.542650,-22.865687,4.971080,-242.344273,145.766810,-61.011816,68.348830,315.436886,-229.959244],
            [3.158110,38.511332,-21.113063,-136.006356,78.501589,-0.386833,66.175944,195.924373,-314.992565,111.042097],
            [1.076598,17.068161,-11.012543,-67.351371,46.292038,-6.762889,30.272241,106.166297,-174.237225,67.831648],
            [-3.885176,29.854859,-58.938856,7.710157,-17.462142,109.378590,-46.263832,-20.007115,26.247429,-91.452346],
            [-0.978470,6.579042,-6.610265,-7.944764,14.936210,-0.860504,-1.763331,-9.116653,-7.617631,1.038812],
            [-4.034509,-37.342861,4.784548,331.039966,-366.395432,205.869561,-73.972454,-783.004822,1266.677170,-601.976188],
            [-1.052115,-2.758087,1.424976,32.085070,-65.625142,42.523742,48.702564,-205.185380,292.368455,-136.435753],
            [-7.607124,33.071252,-6.315526,18.534870,-194.800905,133.812212,-66.872398,87.062831,236.780566,-229.102685],
            [-1.405424,6.335319,0.072628,3.541410,-37.244426,19.865531,-11.205537,11.792846,49.871915,-38.032387],
        ],
        [
            [2.244544,-136.622107,121.222059,662.583482,-53.871373,-564.130711,-782.687281,10.992597,-68.928726,788.052378],
            [2.826728,-126.875987,95.929328,568.394726,-4.136148,-453.847012,-653.000819,-48.994187,-50.280993,623.162830],
            [-3.777337,10.225436,1.240034,92.635366,-51.657646,32.895643,-295.382461,328.650982,-120.179940,-44.357631],
            [-1.447236,5.461403,4.749025,48.008615,-21.845286,2.956496,-158.434140,180.939166,-66.907662,-12.368112],
            [10.042042,-119.568080,79.691020,549.070941,-61.933534,-376.917876,-742.496328,355.223734,-195.563778,507.988585],
            [3.010501,-21.729090,1.991668,112.762690,-16.340675,-23.633676,-177.468164,109.008761,-9.921052,22.807909],
            [0.709180,-170.232986,75.956837,303.253944,95.468513,-465.305031,259.918876,-1021.176731,224.170146,694.309913],
            [-0.719524,-17.830755,11.388082,-11.030840,35.025687,-84.818123,148.163086,-311.586929,90.991650,122.677848],
            [-16.050518,-25.532847,-12.881950,178.634126,4.053477,-63.811942,-203.667997,-106.121629,29.550604,151.992182],
            [-1.673824,-8.875794,1.381913,45.834744,1.856536,-22.438081,-49.265140,-25.587481,7.935613,38.238196],
        ],
        [
            [2.301398,181.303634,-61.869820,-1000.306899,284.737869,159.842412,1142.942260,133.695692,-421.988863,-137.877888],
            [3.987955,149.015264,-60.476105,-797.416901,204.676590,162.295022,908.005087,117.146624,-336.955542,-133.988564],
            [5.031040,-46.142594,19.225997,-110.444474,149.313650,-83.756380,432.341884,-533.799085,191.733096,31.323925],
            [1.725910,-25.685975,13.125859,-47.005471,75.858493,-50.436448,216.654753,-304.871821,113.676236,24.783304],
            [-6.633299,159.585946,15.624555,-814.469438,188.839963,0.432418,1086.383914,-430.704750,40.793702,-37.855211],
            [-1.296914,29.827253,3.937951,-168.567544,25.732007,11.869640,261.134615,-143.554424,19.989276,-13.440427],
            [2.732433,273.480037,-61.775476,-367.557076,-289.783803,266.983017,-507.679600,1763.065563,-776.567300,-135.894460],
            [2.484918,23.321774,-9.828064,68.461375,-93.835371,53.728513,-257.486409,425.892609,-165.187050,-23.208286],
            [-11.233883,40.276201,-8.588129,-313.453505,84.065460,-5.279943,339.141675,217.431003,-266.800686,56.277019],
            [-0.755624,12.390131,-5.113745,-72.540590,15.404470,10.591824,75.270520,46.832565,-54.499727,-1.165841],
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

    @classmethod
    def get_K_matrix(cls, l_l: float, l_r: float) -> np.ndarray:
        K = np.zeros((4, 10))
        for i in range(4):
            for j in range(10):
                K[i, j] = cls.eval_poly33(cls.K_LQR_POLY[i, j], l_l, l_r)
        return K


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
            np.array([1.249406, 0.100938, -26.077168, -8.909536, 32.157903, -24.030688,
                     20.241205, -51.422929, 3.737057, 78.874386]), l_l, l_r)
        
        A[1, 6] = PolynomialFit.eval_poly33(
            np.array([1.303780, -29.689880, 3.228360, 104.251155, -86.025503, -18.350997,
                     -128.701701, 107.769238, 39.182149, 34.614965]), l_l, l_r)
        
        A[3, 4] = PolynomialFit.eval_poly33(
            np.array([10.339915, -8.196393, -191.780470, 4.784725, 65.069958, -222.060167,
                     32.821709, -171.963092, 112.584431, 673.807352]), l_l, l_r)
        
        A[3, 6] = PolynomialFit.eval_poly33(
            np.array([-1.652581, 97.895176, 36.397900, -57.770215, -785.369832, -164.242357,
                     -149.327681, 947.950385, 368.200981, 297.399354]), l_l, l_r)
        
        A[5, 4] = PolynomialFit.eval_poly33(
            np.array([-8.746033, -2.625688, 857.535480, 87.358956, -299.914834, -2690.112160,
                     -163.396025, 343.978909, 173.043928, 2731.910197]), l_l, l_r)
        
        A[5, 6] = PolynomialFit.eval_poly33(
            np.array([-29.314391, 291.020092, 226.612504, -987.406602, 719.436751, -943.092597,
                     1076.631151, -370.375296, -1145.928814, 1227.907928]), l_l, l_r)
        
        A[7, 4] = PolynomialFit.eval_poly33(
            np.array([4.529426, -41.388381, -37.886062, 184.058220, 30.108234, -19.904842,
                     -282.845803, 87.971891, -43.364153, 98.283421]), l_l, l_r)
        
        A[7, 6] = PolynomialFit.eval_poly33(
            np.array([22.334454, 287.368591, 3.232405, -1468.003104, -127.008462, -17.954502,
                     1967.859505, 239.688335, 26.513824, 42.309521]), l_l, l_r)
        
        A[9, 4] = PolynomialFit.eval_poly33(
            np.array([2.400763, 0.193955, -50.107898, -17.119884, 61.792172, -46.175537,
                     38.893956, -98.810377, 7.180844, 151.559004]), l_l, l_r)
        
        A[9, 6] = PolynomialFit.eval_poly33(
            np.array([2.505244, -57.049809, 6.203370, 200.321066, -165.300043, -35.261875,
                     -247.303370, 207.081146, 75.289429, 66.513475]), l_l, l_r)
        
        A[9, 8] = PolynomialFit.eval_poly33(
            np.array([22.172021, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), l_l, l_r)
        
        return A
    
    def _compute_B_matrix(self, l_l: float, l_r: float) -> np.ndarray:
        """计算状态空间B矩阵"""
        B = np.zeros((10, 4))
        
        B[1, 0] = PolynomialFit.eval_poly33(
            np.array([0.631433, 1.202643, 5.602343, -3.299900, 0.037346, -10.023339,
                     3.292600, 0.156717, -0.472287, 5.620443]), l_l, l_r)
        
        B[1, 1] = PolynomialFit.eval_poly33(
            np.array([0.620518, -1.586487, 8.577227, 4.252261, -22.674004, 4.161864,
                     -5.025307, 27.467203, 6.374026, -18.617203]), l_l, l_r)
        
        B[1, 2] = PolynomialFit.eval_poly33(
            np.array([0.021990, 0.219398, -4.035200, -1.128268, 1.978142, 12.493726,
                     1.563654, -1.946008, -1.660046, -12.496103]), l_l, l_r)
        
        B[1, 3] = PolynomialFit.eval_poly33(
            np.array([0.043511, -2.802340, -1.293875, 13.392205, 1.637149, -0.583621,
                     -19.170596, 1.794839, -1.592593, 3.108492]), l_l, l_r)
        
        B[3, 0] = PolynomialFit.eval_poly33(
            np.array([-2.251876, -3.156525, 43.895593, 1.965154, 25.010225, -81.772093,
                     3.009737, -22.790071, -23.050074, 48.177662]), l_l, l_r)
        
        B[3, 1] = PolynomialFit.eval_poly33(
            np.array([-1.675261, 19.645222, 65.120212, -18.613731, -136.818348, 39.460837,
                     -9.288844, 155.165441, 35.202490, -158.727999]), l_l, l_r)
        
        B[3, 2] = PolynomialFit.eval_poly33(
            np.array([0.439465, -1.012892, -32.979275, 0.498761, 8.484486, 104.343521,
                     2.008071, -11.467493, -2.104773, -106.996352]), l_l, l_r)
        
        B[3, 3] = PolynomialFit.eval_poly33(
            np.array([1.757362, -3.120877, -9.696410, 8.875951, 4.429960, -5.611718,
                     -21.222618, 29.022462, -10.497881, 26.488798]), l_l, l_r)
        
        B[5, 0] = PolynomialFit.eval_poly33(
            np.array([-9.933348, -11.594584, -45.455094, 31.031519, 2.471796, 255.207837,
                     -28.025240, -14.893570, 18.443402, -330.712387]), l_l, l_r)
        
        B[5, 1] = PolynomialFit.eval_poly33(
            np.array([-0.049853, 17.732150, -229.801211, -51.723610, 219.351413, 680.888809,
                     39.855492, -145.962589, -276.376880, -639.409728]), l_l, l_r)
        
        B[5, 2] = PolynomialFit.eval_poly33(
            np.array([11.921696, -2.236174, -22.093193, 10.713142, -17.281347, -32.133165,
                     -12.990466, 9.995317, 25.269700, 91.171182]), l_l, l_r)
        
        B[5, 3] = PolynomialFit.eval_poly33(
            np.array([-1.973774, 25.714364, 39.187823, -116.170838, -35.330664, -117.048076,
                     161.003126, 1.725501, 62.816878, 106.504362]), l_l, l_r)
        
        B[7, 0] = PolynomialFit.eval_poly33(
            np.array([-1.660238, -7.364768, 8.031142, 41.633608, -1.001674, -14.798977,
                     -52.659036, -17.094649, 10.700045, 6.920427]), l_l, l_r)
        
        B[7, 1] = PolynomialFit.eval_poly33(
            np.array([-11.951319, 148.072282, 12.251551, -523.203870, -28.805485, 2.927728,
                     625.983629, 5.775070, 18.981947, -23.363482]), l_l, l_r)
        
        B[7, 2] = PolynomialFit.eval_poly33(
            np.array([0.286295, -3.729854, -5.684171, 16.850514, 5.124693, 16.977757,
                     -23.353412, -0.250283, -9.111552, -15.448397]), l_l, l_r)
        
        B[7, 3] = PolynomialFit.eval_poly33(
            np.array([14.050347, -95.040995, -1.884190, 259.444074, 2.282003, -0.368524,
                     -262.983259, 6.473151, -3.763360, 3.917486]), l_l, l_r)
        
        B[9, 0] = PolynomialFit.eval_poly33(
            np.array([-2.129567, 2.310907, 10.765035, -6.340836, 0.071761, -19.260083,
                     6.326809, 0.301135, -0.907511, 10.799815]), l_l, l_r)
        
        B[9, 1] = PolynomialFit.eval_poly33(
            np.array([-2.150541, -3.048472, 16.481346, 8.170820, -43.568636, 7.997121,
                     -9.656247, 52.778882, 12.247842, -35.773396]), l_l, l_r)
        
        B[9, 2] = PolynomialFit.eval_poly33(
            np.array([-7.282144, 0.421579, -7.753732, -2.167993, 3.801046, 24.006991,
                     3.004598, -3.739300, -3.189818, -24.011558]), l_l, l_r)
        
        B[9, 3] = PolynomialFit.eval_poly33(
            np.array([-7.240790, -5.384762, -2.486211, 25.733440, 3.145820, -1.121442,
                     -36.836756, 3.448826, -3.060206, 5.973042]), l_l, l_r)
        
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

        x_ref = np.zeros(10)
        x_ref[self.IDX_S] = float(s_ref)
        x_ref[self.IDX_DS] = float(v_ref)
        x_ref[self.IDX_PHI] = float(theta_ref)
        x_ref[self.IDX_DPHI] = float(omega_ref)

        x_err = self.state - x_ref
        x_err[self.IDX_PHI] = angle_diff(float(self.state[self.IDX_PHI]), float(theta_ref))

        u = -self._cached_K @ x_err

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

        # command targets (10Hz input -> 1000Hz rate-limited application)
        self._last_cmd_time = self.get_clock().now()
        self._spin_slow = False
        self._spin_fast = False

        self._v_target = 0.0
        self._w_target = 0.0
        self._theta_target = float(self.cfg.INIT_THETA)

        self._v_applied = 0.0
        self._w_applied = 0.0

        # s reference
        self._s_ref = float(self.dyn.s)
        self._was_spin_mode = False

        # odom delay buffer
        buffer_sec = max(1.0, float(self.cfg.ODOM_DELAY_SEC) + 1.0)
        self._odom_buf: deque[OdomSample] = deque(maxlen=int(buffer_sec * self.cfg.LQR_FREQ_HZ) + 50)

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

    # ------------------------
    # cmd processing
    # ------------------------

    def _cmd_cb(self, msg: ChassisCmd) -> None:
        now = self.get_clock().now()
        self._last_cmd_time = now

        slow_spin = bool(msg.slow_spin)
        fast_spin = bool(msg.fast_spin)
        self._spin_slow = slow_spin
        self._spin_fast = fast_spin

        if slow_spin or fast_spin:
            # ignore v/theta/omega; spin handled in _on_lqr
            return

        v = float(msg.velocity)
        theta = float(msg.theta)
        w = float(msg.omega)

        if not is_finite(v):
            v = 0.0
        if not is_finite(w):
            w = 0.0
        if not is_finite(theta):
            theta = self._theta_target

        # 目标限幅（速度/角速度 + 乘积）；加速度在 1000Hz 中做 rate-limit
        v, w = self._clamp_v_w_product(v, w)

        self._v_target = v
        self._w_target = w
        self._theta_target = wrap_to_pi(theta)

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

        # apply 1000Hz accel limits
        self._v_applied = self._rate_limit(self._v_applied, self._v_target, self.cfg.MAX_ACCEL, self._dt_lqr)
        self._w_applied = self._rate_limit(self._w_applied, self._w_target, self.cfg.MAX_ANG_ACCEL, self._dt_lqr)

        # enforce max speed / max omega / product after rate-limit
        self._v_applied, self._w_applied = self._clamp_v_w_product(self._v_applied, self._w_applied)

        # update theta reference
        if spin_mode:
            self._theta_target = wrap_to_pi(self._theta_target + self._w_applied * self._dt_lqr)

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
        msg.leg_mode = 4
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
        radius = 0.3
        period = 10.0
        cx = 6.5
        cz = 0.6
        cy = 6.0 + 0.5 * math.sin(2.0 * math.pi * now.nanoseconds / 1e9 / period)

        # sample sphere surface with grid in spherical coordinates
        num_theta = 40
        num_phi = 40
        points = []
        for i in range(num_phi):
            phi = math.pi * (i + 0.5) / num_phi
            for j in range(num_theta):
                theta = 2.0 * math.pi * j / num_theta
                px = cx + radius * math.sin(phi) * math.cos(theta)
                py = cy + radius * math.sin(phi) * math.sin(theta)
                pz = cz + radius * math.cos(phi)
                points.append((px, py, pz))

        msg = PointCloud2()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "odom"
        msg.height = 1
        msg.width = len(points)

        # fields: x, y, z as float32
        msg.fields = []
        msg.fields.append(PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1))
        msg.fields.append(PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1))
        msg.fields.append(PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1))

        msg.is_bigendian = False
        msg.point_step = 12  # 3 * 4 bytes
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True

        # pack points as float32 little-endian
        data = bytearray()
        for (px, py, pz) in points:
            data += struct.pack('<fff', float(px), float(py), float(pz))

        msg.data = bytes(data)
        self.cloud_pub.publish(msg)

    def _pub_tf(self) -> None:
        now = self.get_clock().now()
        tf = TransformStamped()
        tf.header.stamp = now.to_msg()
        tf.header.frame_id = self.cfg.FRAME_ODOM
        tf.child_frame_id = "map"
        tf.transform.translation.x = 0.0
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.0
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
