import numpy as np
from scipy.spatial.transform import Rotation as R

def se3_inverse(xyz, xyzw):
    """
    SE(3) 位姿求逆
    :param xyz: 位置 [x, y, z]
    :param xyzw: 四元数 [x, y, z, w] (注意：scipy 使用 xyzw 顺序)
    :return: (xyz_inv, xyzw_inv)
    """
    p = np.array(xyz)
    q = np.array(xyzw)
    
    # 旋转部分：四元数共轭（单位四元数的逆）
    q_inv = R.from_quat(q).inv().as_quat()  # 返回 [x, y, z, w]
    
    # 平移部分：p_inv = -R^T @ p
    R_mat = R.from_quat(q).as_matrix()
    p_inv = -R_mat.T @ p
    
    return p_inv, q_inv

# ===== 示例 =====
xyz = list(map(float, input("输入位置 (x y z): ").split()))
xyzw = list(map(float, input("输入四元数 (x y z w): ").split()))

xyz_inv, xyzw_inv = se3_inverse(xyz, xyzw)
print("逆位姿位置:", xyz_inv)
print("逆位姿四元数:", xyzw_inv)