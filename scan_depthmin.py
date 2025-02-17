from tqdm import tqdm  # Import tqdm để hiển thị tiến trình
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os

name_datasets = "datasets_nyu+SUN-RGBD-2D"
depth_path = f"./{name_datasets}/depth_maps"

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")  # Các định dạng ảnh hợp lệ

image_files = [f for f in os.listdir(depth_path) if f.lower().endswith(image_extensions)]

# Kiểm tra nếu có ảnh trong thư mục
if not image_files:
    print("Không tìm thấy ảnh nào trong thư mục!")
else:
    # Danh sách chứa giá trị min và max của từng ảnh
    depth_min_values = []
    depth_max_values = []
    for image_file in tqdm(image_files, desc="Đang xử lý ảnh", unit="ảnh"):
        # Chọn ngẫu nhiên một ảnh
        image_path = os.path.join(depth_path, image_file)

        # Đọc ảnh bằng OpenCV
        depth = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        depth = cv2.resize(depth, (384, 384), interpolation=cv2.INTER_LINEAR)
        depth = depth / 255.0  # Chuẩn hóa về 0-1
        # depth = np.clip(depth, 1e-6, 1.0)  # Tránh giá trị 0 tuyệt đối
        # print(depth.min())
        # Lưu giá trị min và max vào danh sách
        depth_min_values.append(depth.min())
        depth_max_values.append(depth.max())

# Vẽ biểu đồ thống kê các giá trị depth.min() và depth.max()
plt.figure(figsize=(14, 6))

# Biểu đồ depth.min()
plt.subplot(1, 2, 1)
plt.hist(depth_min_values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Thống kê giá trị depth.min()', fontsize=14)
plt.xlabel('Giá trị depth.min()', fontsize=12)
plt.ylabel('Số lượng ảnh', fontsize=12)
plt.grid(True)

# Biểu đồ depth.max()
plt.subplot(1, 2, 2)
plt.hist(depth_max_values, bins=30, color='salmon', edgecolor='black', alpha=0.7)
plt.title('Thống kê giá trị depth.max()', fontsize=14)
plt.xlabel('Giá trị depth.max()', fontsize=12)
plt.ylabel('Số lượng ảnh', fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.show()
