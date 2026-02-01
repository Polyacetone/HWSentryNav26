"""LQR重线性化与离散化工具（10Hz）

从 whole_body_follow_sim.py 读取连续 A/B/K，多项式取值于给定腿长，
使用零阶保持将 (A,B) 离散化到给定 dt，然后输出到 YAML。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy.linalg import expm, solve_discrete_are

@dataclass
class RobotParams:
    """机器人物理参数"""
    # 采样时间
    Ts: float = 0.001
    
    # 几何参数
    R_w: float = 0.12 / 2       # 驱动轮半径
    R_l: float = 0.442 / 2      # 驱动轮轮距/2
    
    # 腿长参数 (可变)
    l_l: float = 0.15           # 左腿长
    l_r: float = 0.15           # 右腿长
    h_l: float = 0.15           # 左腿高度
    h_r: float = 0.15           # 右腿高度
    
    # 质心参数
    l_c: float = 0.04275        # 机体质心到腿部关节连线距离
    
    # 质量参数
    m_w: float = 0.31509        # 驱动轮质量
    m_l: float = 3.38608        # 腿部质量
    m_b: float = 9.59766        # 机体质量
    
    # 惯量参数
    I_w: float = 0.0050935      # 驱动轮转动惯量
    I_l_l: float = 0.02155      # 左腿转动惯量
    I_l_r: float = 0.02155      # 右腿转动惯量
    I_b: float = 1.0429         # 机体转动惯量
    I_z: float = 0.495599       # 机器人z轴转动惯量
    
    # 重力加速度
    g: float = 9.7936
    
    # 腿长范围
    h_min: float = 0.10
    h_max: float = 0.33
    
    # 轮电机最大力矩 (Nm)
    T_w_max: float = 20.0
    # 腿电机最大力矩 (Nm)
    T_b_max: float = 30.0
    
    @property
    def l_w_l(self) -> float:
        """驱动轮到左腿部质心距离"""
        return self.l_l
    
    @property
    def l_w_r(self) -> float:
        """驱动轮到右腿部质心距离"""
        return self.l_r
    
    @property
    def l_b_l(self) -> float:
        """驱动轮到左腿身体部质心距离"""
        return self.l_l / 2
    
    @property
    def l_b_r(self) -> float:
        """驱动轮到右腿身体部质心距离"""
        return self.l_r / 2


# ============================================================================
# 多项式拟合类 (用于计算随腿长变化的LQR增益矩阵)
# ============================================================================

class PolynomialFit:
    """多项式拟合类，用于计算随腿长变化的LQR增益矩阵"""
    
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
        """计算poly33多项式的值
        
        poly33: p00 + p10*x + p01*y + p20*x^2 + p11*x*y + p02*y^2 + 
                p30*x^3 + p21*x^2*y + p12*x*y^2 + p03*y^3
        
        coeffs顺序: [p00, p10, p01, p20, p11, p02, p30, p21, p12, p03]
        """
        return (coeffs[0] + 
                coeffs[1] * x + coeffs[2] * y +
                coeffs[3] * x**2 + coeffs[4] * x * y + coeffs[5] * y**2 +
                coeffs[6] * x**3 + coeffs[7] * x**2 * y + 
                coeffs[8] * x * y**2 + coeffs[9] * y**3)
    
    @classmethod
    def get_K_matrix(cls, l_l: float, l_r: float) -> np.ndarray:
        """根据左右腿长计算LQR增益矩阵K (4x10)"""
        K = np.zeros((4, 10))
        for i in range(4):
            for j in range(10):
                K[i, j] = cls.eval_poly33(cls.K_LQR_POLY[i, j], l_l, l_r)
        return K


# ============================================================================
# 轮腿平衡机器人动力学模型
# ============================================================================

class WheelLegDynamics:
    """轮腿平衡机器人动力学模型"""
    
    # 状态索引
    IDX_S = 0           # 水平位移
    IDX_DS = 1          # 水平速度
    IDX_PHI = 2         # 偏航角
    IDX_DPHI = 3        # 偏航角速度
    IDX_THETA_L_L = 4   # 左腿倾斜角
    IDX_DTHETA_L_L = 5  # 左腿倾斜角速度
    IDX_THETA_L_R = 6   # 右腿倾斜角
    IDX_DTHETA_L_R = 7  # 右腿倾斜角速度
    IDX_THETA_B = 8     # 机体俯仰角
    IDX_DTHETA_B = 9    # 机体俯仰角速度
    
    def __init__(self, params: Optional[RobotParams] = None):
        """初始化动力学模型"""
        self.params = params if params is not None else RobotParams()
        
        # 状态向量 [s, ds, phi, dphi, theta_l_l, dtheta_l_l, theta_l_r, dtheta_l_r, theta_b, dtheta_b]
        self.state = np.zeros(10)
        
        # 世界坐标系位置 [x, y, theta]
        self.world_pose = np.zeros(3)
        
        # 缓存A和B矩阵
        self._cached_l_l = None
        self._cached_l_r = None
        self._cached_A = None
        self._cached_B = None
        self._cached_K = None
    
    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """重置机器人状态"""
        self.state = np.zeros(10)
        self.world_pose = np.array([x, y, theta])
        self.state[self.IDX_PHI] = theta
    
    def _update_cache(self, l_l: float, l_r: float):
        """更新缓存的矩阵"""
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
    
    def compute_lqr_control(self, v_ref: float, w_ref: float) -> np.ndarray:
        """计算LQR控制输入
        
        Args:
            v_ref: 参考线速度 (m/s)
            w_ref: 参考角速度 (rad/s)
        
        Returns:
            u: 控制输入 [T_w_l, T_w_r, T_b_l, T_b_r]
        """
        # 获取当前腿长
        l_l = self.params.l_l
        l_r = self.params.l_r
        
        # 限制腿长在有效范围内
        l_l = np.clip(l_l, self.params.h_min, self.params.h_max)
        l_r = np.clip(l_r, self.params.h_min, self.params.h_max)
        
        # 更新缓存
        self._update_cache(l_l, l_r)
        
        # 构建参考状态
        x_ref = np.zeros(10)
        x_ref[self.IDX_S] = self.state[self.IDX_S]  # 位置参考跟随当前位置
        x_ref[self.IDX_DS] = v_ref                   # 速度参考
        x_ref[self.IDX_PHI] = self.state[self.IDX_PHI]  # 角度参考跟随当前角度
        x_ref[self.IDX_DPHI] = w_ref                 # 角速度参考
        
        # 计算状态误差
        x_error = self.state - x_ref
        
        # 计算控制输入 u = -K * x_error
        u = -self._cached_K @ x_error
        
        # 力矩限幅
        u[0] = np.clip(u[0], -self.params.T_w_max, self.params.T_w_max)
        u[1] = np.clip(u[1], -self.params.T_w_max, self.params.T_w_max)
        u[2] = np.clip(u[2], -self.params.T_b_max, self.params.T_b_max)
        u[3] = np.clip(u[3], -self.params.T_b_max, self.params.T_b_max)
        
        return u
    
    def step(self, v_ref: float, w_ref: float, dt: float) -> np.ndarray:
        """执行一步仿真
        
        Args:
            v_ref: 参考线速度 (m/s)
            w_ref: 参考角速度 (rad/s)
            dt: 时间步长
        
        Returns:
            u: 控制输入 [T_w_l, T_w_r, T_b_l, T_b_r]
        """
        # 计算LQR控制
        u = self.compute_lqr_control(v_ref, w_ref)
        
        # 获取当前腿长
        l_l = np.clip(self.params.l_l, self.params.h_min, self.params.h_max)
        l_r = np.clip(self.params.l_r, self.params.h_min, self.params.h_max)
        
        # 更新缓存
        self._update_cache(l_l, l_r)
        
        # 状态更新 (欧拉积分)
        x_dot = self._cached_A @ self.state + self._cached_B @ u
        self.state = self.state + x_dot * dt
        
        # 更新世界坐标系位置
        v = self.state[self.IDX_DS]
        theta = self.world_pose[2]
        
        self.world_pose[0] += v * math.cos(theta) * dt  # x
        self.world_pose[1] += v * math.sin(theta) * dt  # y
        self.world_pose[2] = self.state[self.IDX_PHI]   # theta
        
        return u
    
    @property
    def velocity(self) -> float:
        """当前线速度"""
        return self.state[self.IDX_DS]
    
    @property
    def palstance(self) -> float:
        """当前角速度"""
        return self.state[self.IDX_DPHI]
    
    @property
    def pitch(self) -> float:
        """当前俯仰角"""
        return self.state[self.IDX_THETA_B]
    
    @property
    def x(self) -> float:
        """当前x坐标"""
        return self.world_pose[0]
    
    @property
    def y(self) -> float:
        """当前y坐标"""
        return self.world_pose[1]
    
    @property
    def theta(self) -> float:
        """当前航向角"""
        return self.world_pose[2]


def discretize_ab(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order hold 离散化。"""
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m), dtype=float)
    M[:n, :n] = A
    M[:n, n:] = B
    Mexp = expm(M * dt)
    Ad = Mexp[:n, :n]
    Bd = Mexp[:n, n:]
    return Ad, Bd


def compute_discrete_lqr_gain(Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """使用离散代数 Riccati 方程 (DARE) 计算离散 LQR 增益 Kd。
    
    对于离散系统 x_{k+1} = Ad * x_k + Bd * u_k，
    最优控制律为 u_k = -Kd * x_k。
    
    Args:
        Ad: 离散化后的状态矩阵 (n x n)
        Bd: 离散化后的输入矩阵 (n x m)
        Q: 状态代价矩阵 (n x n)
        R: 输入代价矩阵 (m x m)
    
    Returns:
        Kd: 离散 LQR 增益矩阵 (m x n)
    """
    # 求解离散代数 Riccati 方程: Ad.T @ P @ Ad - P - Ad.T @ P @ Bd @ inv(R + Bd.T @ P @ Bd) @ Bd.T @ P @ Ad + Q = 0
    P = solve_discrete_are(Ad, Bd, Q, R)
    # Kd = inv(R + Bd.T @ P @ Bd) @ Bd.T @ P @ Ad
    Kd = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return Kd


def flatten_row_major(mat: np.ndarray) -> list[float]:
    return [float(x) for x in mat.reshape(-1)]


def main() -> int:
    parser = argparse.ArgumentParser(description="LQR重线性化/离散化（10Hz）")
    parser.add_argument("--leg-left", type=float, default=0.15, help="左腿长度 (m)")
    parser.add_argument("--leg-right", type=float, default=0.15, help="右腿长度 (m)")
    parser.add_argument("--dt", type=float, default=0.1, help="MPC步长 (s)")
    parser.add_argument("--substeps", type=int, default=10, help="每个MPC步内的子步数（用于更精确的闭环离散化）")
    parser.add_argument("--torque-w-max", type=float, default=20.0, help="轮电机力矩限幅")
    parser.add_argument("--torque-b-max", type=float, default=30.0, help="腿电机力矩限幅")
    parser.add_argument("--output", type=str, default="-", help="输出文件路径，- 表示 stdout")
    parser.add_argument("--format", type=str, default="yaml", choices=["yaml", "json"], help="输出格式")
    args = parser.parse_args()

    dyn = WheelLegDynamics()
    A = dyn._compute_A_matrix(args.leg_left, args.leg_right)
    B = dyn._compute_B_matrix(args.leg_left, args.leg_right)
    K = PolynomialFit.get_K_matrix(args.leg_left, args.leg_right)

    # 计算子步长
    dt_sub = args.dt / args.substeps
    
    # 计算连续时间闭环矩阵: A_cl = A - B @ K
    A_cl = A - B @ K
    n = A_cl.shape[0]
    BK = B @ K
    
    # 构建增广系统用于离散化闭环动力学
    # 闭环系统: dx/dt = A_cl * x + B @ K * x_ref
    # 使用 ZOH 离散化（假设 x_ref 在子步内恒定）
    M = np.zeros((n + n, n + n), dtype=float)
    M[:n, :n] = A_cl
    M[:n, n:] = BK
    
    # 计算子步的离散化矩阵
    Mexp_sub = expm(M * dt_sub)
    A_cl_sub = Mexp_sub[:n, :n]
    B_ref_sub = Mexp_sub[:n, n:]
    
    # 同时输出原始的离散化 A, B 用于备用
    Ad, Bd = discretize_ab(A, B, args.dt)

    payload = {
        "mpc": {
            "lqr": {
                "torque_w_max": float(args.torque_w_max),
                "torque_b_max": float(args.torque_b_max),
                "substeps": args.substeps,
                "A_cl": flatten_row_major(A_cl_sub),
                "B_ref": flatten_row_major(B_ref_sub),
                "A": flatten_row_major(Ad),
                "B": flatten_row_major(Bd),
                "K": flatten_row_major(K),
            }
        }
    }

    if args.format == "json":
        content = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        # 简单 YAML 输出（不依赖外部库）
        def to_yaml_list(xs: list[float]) -> str:
            return "[" + ", ".join(f"{x:.8f}" for x in xs) + "]"

        lqr = payload["mpc"]["lqr"]
        content = (
            "mpc:\n"
            "  lqr:\n"
            f"    torque_w_max: {lqr['torque_w_max']}\n"
            f"    torque_b_max: {lqr['torque_b_max']}\n"
            f"    substeps: {lqr['substeps']}\n"
            f"    A_cl: {to_yaml_list(lqr['A_cl'])}\n"
            f"    B_ref: {to_yaml_list(lqr['B_ref'])}\n"
            f"    A: {to_yaml_list(lqr['A'])}\n"
            f"    B: {to_yaml_list(lqr['B'])}\n"
            f"    K: {to_yaml_list(lqr['K'])}\n"
        )

    if args.output == "-":
        print(content)
    else:
        Path(args.output).write_text(content, encoding="utf-8")
        print(f"[OK] 写入: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
