import os
import cv2
import sys
import time
import torch
import psutil
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image, ImageGrab
from torchvision import transforms
from torch.cuda.amp import autocast  # Import autocast nếu cần
from model import DepthVisionTransformer
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates

# Hàm lấy mức sử dụng GPU
def get_gpu_usage():
    try:
        handle = nvmlDeviceGetHandleByIndex(0)  # Lấy GPU đầu tiên
        utilization = nvmlDeviceGetUtilizationRates(handle)
        return utilization.gpu  # % sử dụng GPU
    except:
        return 0  # Nếu không có GPU, trả về 0

# Hàm in dòng xóa ký tự dư thừa
def print_dynamic(content):
    sys.stdout.write(f"\r{' ' * 100}\r{content}")
    sys.stdout.flush()

def main():
    # Biến thời gian
    last_time = time.time()
    last_time_chip = time.time()
    update_interval = 1  # Thời gian cập nhật (giây)
    cpu_usage, gpu_usage = 0, 0  # Khởi tạo giá trị CPU và GPU
    # Thiết lập thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (f"Using {device}")

    # Kiểm tra card đồ họa Nvidia 
    try:
        nvmlInit()  # Khởi tạo NVML
    except:
        print("Không có đồ họa Nvidia")

    # Khởi tạo mô hình Depth Vision Transformer và đưa lên thiết bị
    model = DepthVisionTransformer().to(device)
    model.eval()  # Chế độ inference
    
    # Vòng lặp chính
    while True:
        # Lấy ảnh màn hình
        original_image = np.array(ImageGrab.grab(bbox=(0, 25, 800, 625)))

        # Resize ảnh về 384x384
        img_cv_resized = cv2.resize(original_image, (384, 384))

        # Định nghĩa transform: chuyển ảnh thành tensor (giá trị [0,1])
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Thêm batch dimension: [1, 3, 384, 384]
        img_tensor = transform(img_cv_resized).unsqueeze(0).to(device)

        process_time_chip = time.time() - last_time_chip
        # Lấy giá trị chipset sau update_interval
        if process_time_chip >= update_interval:
            last_time_chip = time.time()
            cpu_usage = psutil.cpu_percent(interval=0)  # % CPU
            gpu_usage = get_gpu_usage()  # % GPU
        
        # Đồng bộ nếu dùng GPU
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Đo thời gian xử lý mô hình
        start_time_model = time.time()

        # Dự đoán depth map
        with torch.no_grad():
            predicted_depth = model(img_tensor)

        # Chuyển depth map về numpy, loại bỏ batch và channel nếu cần
        depth_map = predicted_depth.squeeze().cpu().numpy()

        # Chuẩn hóa depth map về khoảng 0-255 để hiển thị (uint8)
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = np.uint8(depth_norm)
        
        # Áp dụng colormap để trực quan hóa depth map
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time_model = time.time()

        processing_time = end_time_model - start_time_model

        # Hiển thị ảnh đầu vào và đầu ra
        cv2.imshow('Original Image', cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))  # Hiển thị ảnh đầu vào
        cv2.imshow('Processed Depth Map Color', depth_color)  # Hiển thị ảnh đầu ra với colormap
        
        # Tính thời gian xử lý
        process_time = time.time() - last_time
        last_time = time.time()
        content = f"Look took {process_time:.2f} s | Model Processing Time: {processing_time:.4f} s | CPU: {cpu_usage}% | GPU: {gpu_usage}%"
        print_dynamic(content)

        # Dừng chương trình khi nhấn 'q'
        if cv2.waitKey(25) == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()