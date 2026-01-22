import cv2
import numpy as np
import math

def generate_navigation_map(
    height_map_path, 
    step_threshold=5,        # 台阶阈值（高程差单位）
    max_obstacle_height=20,  # 机器人最大越障高度
    cost_coeff=10            # 代价系数
):
    # 1. 读取高程图（灰度图）
    height_map = cv2.imread(height_map_path, cv2.IMREAD_GRAYSCALE)
    if height_map is None:
        raise ValueError(f"无法读取图像: {height_map_path}")
    
    # 转为有符号整型，避免减法时 uint8 溢出
    height_map_int = height_map.astype(np.int32)

    h, w = height_map.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)  # 三通道输出图 [B,G,R] -> [向量X, 向量Y, 代价]
    
    # --- 优化开始：使用矩阵运算代替双重循环，并引入梯度场平滑 ---
    
    # 2. 计算高程差 (Cost) - 使用形态学膨胀
    # 膨胀操作只支持 uint8 或 float32
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(height_map, kernel)  # 用原始 uint8 图像
    dilated_int = dilated.astype(np.int32)
    # 局部最大高程差 = 邻域最大值 - 当前值 (只考虑上升)
    diff_map = dilated_int - height_map_int
    
    # 3. 计算方向 (Direction) - 使用 Sobel + Gaussian 平滑
    # 原始方法只看单点，容易受噪点影响。
    # 这里先用 Sobel 算子计算梯度 (本身包含3x3邻域信息)
    gx = cv2.Sobel(height_map, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(height_map, cv2.CV_64F, 0, 1, ksize=3)
    
    # 关键优化：对梯度场进行高斯平滑
    # 这相当于参考了周围的点。由于台阶边缘是线性的，
    # 沿着台阶方向的梯度方向是一致的，平滑操作会增强主方向，消除随机噪点。
    # 这样得到的法向量既准又稳，且不会像单纯增大卷积核那样导致边缘过度膨胀。
    gx = cv2.GaussianBlur(gx, (7, 7), 0)
    gy = cv2.GaussianBlur(gy, (7, 7), 0)
    
    # 计算模长并归一化
    magnitude = np.sqrt(gx**2 + gy**2)
    magnitude[magnitude == 0] = 1.0  # 避免除零
    
    unit_x = gx / magnitude
    unit_y = gy / magnitude
    
    # 映射向量到 [1, 255], 128为中心
    vec_x_map = (unit_x * 127 + 128).astype(np.uint8)
    vec_x_map.clip(1, 255, out=vec_x_map)  # 避免0值
    vec_y_map = (unit_y * 127 + 128).astype(np.uint8)
    vec_y_map.clip(1, 255, out=vec_y_map)  # 避免0值
    
    # 4. 生成输出图 (向量化操作)
    # Mask 1: 不可跨越障碍 (高程差 >= 最大越障能力)
    mask_obstacle = diff_map >= max_obstacle_height
    output[mask_obstacle, 2] = 255
    
    # Mask 2: 可跨越台阶 (台阶阈值 <= 高程差 < 最大越障能力)
    mask_step = (diff_map >= step_threshold) & (diff_map < max_obstacle_height)
    
    # 计算代价
    raw_cost = diff_map * cost_coeff
    cost_vals = np.clip(raw_cost, 0, 254).astype(np.uint8)
    
    # 赋值
    output[mask_step, 2] = cost_vals[mask_step]
    output[mask_step, 0] = vec_x_map[mask_step]
    output[mask_step, 1] = vec_y_map[mask_step]
    
    # --- 优化结束 ---
    
    # 4. 填充被255包围的内部区域（封闭空洞）
    cost_map = output[:, :, 2].copy()  # 当前代价图，255 = 不可通行

    # 创建 mask (size h+2, w+2)，用于 floodFill
    # 注意：OpenCV 要求 mask 边界为 0，内部障碍设为 1（非零），不能用 255！
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    mask[1:h+1, 1:w+1] = (cost_map == 255).astype(np.uint8)  # 障碍区域设为 1

    # 创建一个临时图像用于 floodFill（必须是非 const，会被修改）
    # 我们用全 0 图像，只关心 mask 控制的连通性
    external_marker = np.zeros((h, w), dtype=np.uint8)

    # 从四个角开始泛洪（只要该点不是障碍）
    seeds = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    for sy, sx in seeds:
        # 注意：OpenCV floodFill seedPoint 是 (x, y)，即 (col, row)
        if cost_map[sy, sx] < 255 and external_marker[sy, sx] == 0:
            # flags: 4-connected, 且在 mask 上标记填充区域
            cv2.floodFill(
                image=external_marker,
                mask=mask,
                seedPoint=(sx, sy),
                newVal=255,           # 把 external_marker 中的区域标为 255
                loDiff=0,
                upDiff=0,
                flags=4 | (255 << 8)  # 4-连通，填充值 255
            )

    # 现在 external_marker 中值为 255 的区域 = 从边界可达的非障碍区域
    # 内部封闭区域 = 非障碍 & 未被标记
    internal_region = (cost_map < 255) & (external_marker != 255)

    # 将这些内部封闭区域设为不可通行
    # output[internal_region, 2] = 255  # 代价
    # output[internal_region, 0] = 0    # 向量X
    # output[internal_region, 1] = 0    # 向量Y
    
    return output

# 使用示例
if __name__ == "__main__":
    # 参数设置 (根据机器人性能调整)
    STEP_THRESHOLD = 10       # 台阶阈值
    MAX_HEIGHT = 40         # 机器人最大越障高度
    COST_COEFF = 0.5          # 代价系数
    
    # 生成导航地图
    navigation_map = generate_navigation_map(
        height_map_path="height_map.png",
        step_threshold=STEP_THRESHOLD,
        max_obstacle_height=MAX_HEIGHT,
        cost_coeff=COST_COEFF
    )
    
    # 保存结果
    cv2.imwrite("navigation_map.png", navigation_map)