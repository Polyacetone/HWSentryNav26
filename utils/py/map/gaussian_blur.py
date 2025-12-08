import cv2
import numpy as np

def process_image(image_path, output_path):
    # 读取彩色PNG文件
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # 提取第三个通道
    third_channel = image[:, :, 2]

    # 对膨胀后的通道进行高斯模糊
    blurred_channel = cv2.GaussianBlur(src=third_channel, ksize=(5, 5), sigmaX=0, sigmaY=0)

    # 将处理后的通道放回原图
    processed_image = image.copy()
    processed_image[:, :, 2] = blurred_channel

    # 保存结果
    cv2.imwrite(output_path, processed_image)

if __name__ == "__main__":
    input_path = "navigation_map_edited.png"  # 输入文件路径
    output_path = "navigation_map_dilated.png"  # 输出文件路径
    process_image(input_path, output_path)