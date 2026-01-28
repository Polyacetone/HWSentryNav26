import numpy as np
import cv2, os
from scipy.ndimage import distance_transform_edt

def inflate_navigation_map(img, robot_radius, cutoff_radius, decay_alpha=0.5):
    """
    高精度离线膨胀导航地图
    :param img: 原始BGR地图 (H, W, 3)
    :param robot_radius: 机器人半径 (pixel)
    :param cutoff_radius: 截断半径 (pixel)
    :param decay_alpha: 指数衰减系数
    :return: 膨胀后的BGR地图
    """
    h, w = img.shape[:2]
    
    # --- 1. 数据预处理 ---
    # 分离通道
    b_chan = img[:, :, 0].astype(np.float32)
    g_chan = img[:, :, 1].astype(np.float32)
    r_chan = img[:, :, 2].astype(np.float32)

    # 障碍物掩码
    obs_mask = (r_chan == 255)
    
    # 台阶方向场源提取: 只有(B,G)不全为0的才是源
    step_source_mask = (b_chan != 0) | (g_chan != 0)
    
    # 将BG映射为向量场 (-127 to 127)
    # 128映射为0
    vx_field = b_chan - 128
    vy_field = g_chan - 128
    # 强制将非源区域清零
    vx_field[~step_source_mask] = 0
    vy_field[~step_source_mask] = 0

    # --- 2. 障碍物通道膨胀 (R) ---
    # 计算每个像素到最近障碍物的距离
    dist_to_obs = distance_transform_edt(~obs_mask)
    
    inflated_r = np.zeros((h, w), dtype=np.float32)
    # 机器人半径内
    inflated_r[dist_to_obs <= robot_radius] = 255
    # 半径外截断范围内指数衰减
    decay_range = (dist_to_obs > robot_radius) & (dist_to_obs <= cutoff_radius)
    # cost = 255 * exp(-alpha * (d - d_robot))
    inflated_r[decay_range] = 255 * np.exp(-decay_alpha * (dist_to_obs[decay_range] - robot_radius))

    # --- 3. 台阶方向场膨胀 (BG) ---
    # 结果缓冲区
    res_vx = np.zeros((h, w), dtype=np.float32)
    res_vy = np.zeros((h, w), dtype=np.float32)
    res_max_mag = np.zeros((h, w), dtype=np.float32)
    # 标记哪些位置被膨胀信号覆盖过（用于区分(0,0)和(128,128)）
    is_covered = np.zeros((h, w), dtype=bool)

    # 提取所有源像素坐标
    src_ys, src_xs = np.where(step_source_mask)
    
    # 为了保证绝对精确，对每个源进行传播（离线处理不考虑效率）
    # 如果地图极大，可以考虑使用kd-tree优化邻域搜索
    for sy, sx in zip(src_ys, src_xs):
        v0_x = vx_field[sy, sx]
        v0_y = vy_field[sy, sx]
        v0_mag = np.sqrt(v0_x**2 + v0_y**2)
        
        # 确定影响范围（矩形包围盒优化）
        y_min, y_max = max(0, sy - cutoff_radius), min(h, sy + cutoff_radius + 1)
        x_min, x_max = max(0, sx - cutoff_radius), min(w, sx + cutoff_radius + 1)
        
        # 计算该源到范围内所有点的距离
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        dist_sq = (yy - sy)**2 + (xx - sx)**2
        dist = np.sqrt(dist_sq)
        
        # 掩码：在截断半径内
        mask = dist <= cutoff_radius
        
        # 计算在该点的衰减后的向量模长
        # 规则：R_robot内模长不变，R_robot外指数衰减
        m_dist = dist[mask]
        current_mag = np.full_like(m_dist, v0_mag)
        decay_idx = m_dist > robot_radius
        current_mag[decay_idx] *= np.exp(-decay_alpha * (m_dist[decay_idx] - robot_radius))
        
        # 计算衰减后的向量
        # 注意：方向保持源的方向
        ratio = current_mag / v0_mag
        curr_vx = v0_x * ratio
        curr_vy = v0_y * ratio
        
        # 更新缓冲区
        target_res_vx = res_vx[y_min:y_max, x_min:x_max]
        target_res_vy = res_vy[y_min:y_max, x_min:x_max]
        target_max_mag = res_max_mag[y_min:y_max, x_min:x_max]
        target_covered = is_covered[y_min:y_max, x_min:x_max]
        
        target_res_vx[mask] += curr_vx
        target_res_vy[mask] += curr_vy
        target_max_mag[mask] = np.maximum(target_max_mag[mask], current_mag)
        target_covered[mask] = True

    # 方向场归一化与重新缩放
    total_mag = np.sqrt(res_vx**2 + res_vy**2)
    # 避免除以0
    valid_mag = total_mag > 1e-6
    
    final_vx = np.zeros_like(res_vx)
    final_vy = np.zeros_like(res_vy)
    
    # 核心公式：V_final = (Sum_V / |Sum_V|) * Max_Individual_Mag
    final_vx[valid_mag] = (res_vx[valid_mag] / total_mag[valid_mag]) * res_max_mag[valid_mag]
    final_vy[valid_mag] = (res_vy[valid_mag] / total_mag[valid_mag]) * res_max_mag[valid_mag]

    # --- 4. 映射回BGR图像 ---
    out_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # R通道：障碍物
    out_img[:, :, 2] = np.clip(inflated_r, 0, 255).astype(np.uint8)
    
    # BG通道：方向场
    # 如果 is_covered 为 False，保持 0 (即原图(0,0))
    # 如果 is_covered 为 True，映射到 [1, 255]，中心 128
    final_b = np.zeros((h, w), dtype=np.float32)
    final_g = np.zeros((h, w), dtype=np.float32)
    
    # 线性映射：[-127, 127] -> [1, 255]
    final_b[is_covered] = final_vx[is_covered] + 128
    final_g[is_covered] = final_vy[is_covered] + 128
    
    # 限制范围并处理 (128,128) 逻辑
    # 即使模长非常小，加了128后也会在128左右，不会变成0
    out_img[:, :, 0] = np.clip(final_b, 1, 255).astype(np.uint8)
    out_img[:, :, 1] = np.clip(final_g, 1, 255).astype(np.uint8)
    
    # 对于从未被覆盖的区域，强制回填为0 (虽然上面初始化为0了，这里做个强调)
    out_img[~is_covered, 0] = 0
    out_img[~is_covered, 1] = 0

    # 对台阶方向场模长非常小的区域，设置方向为(0,0)，同时根据方向模长设置障碍物以防误导
    # small_mag_mask = np.sqrt(final_vx**2 + final_vy**2) < 20
    # out_img[small_mag_mask, 0] = 0
    # out_img[small_mag_mask, 1] = 0
    # out_img[small_mag_mask, 2] = np.maximum(
    #     out_img[small_mag_mask, 2],
    #     (np.sqrt(final_vx**2 + final_vy**2)[small_mag_mask] * 0.8).astype(np.uint8)
    # )

    return out_img

if __name__ == "__main__":
    map_path = input("输入原始导航地图路径: ")
    save_path = os.path.splitext(map_path)[0] + "_inflated.png"
    original_map = cv2.imread(map_path)
    robot_radius = 1  # 像素
    cutoff_radius = 4  # 像素
    decay_alpha = 1.0  # 衰减系数
    inflated_map = inflate_navigation_map(original_map, robot_radius, cutoff_radius, decay_alpha)
    cv2.imwrite(save_path, inflated_map)