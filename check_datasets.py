import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from PIL import Image
from model import DepthVisionTransformer

class SILogLoss(nn.Module):
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None):
        # Nếu có mask thì lấy những phần tử tương ứng
        if mask is not None:
            input = input[mask]
            target = target[mask]
        
        # Tránh giá trị bằng 0
        input = torch.clamp(input, min=1e-6)
        target = torch.clamp(target, min=1e-6)
        
        # Tính toán SILog loss
        g = torch.log(input) - torch.log(target)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)

class DepthDataset(Dataset):
    def __init__(self, image_dir, depth_dir, transform=None, target_size=(384,384)):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.depth_filenames = sorted(os.listdir(depth_dir))
        self.transform = transform
        self.target_size = target_size  # Kích thước mong muốn cho cả ảnh và depth map

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Đọc ảnh RGB
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Đọc và xử lý depth map
        depth_path = os.path.join(self.depth_dir, self.depth_filenames[idx])
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        depth = depth / 255.0           # Chuẩn hóa về 0-1
        depth = np.clip(depth, 1e-6, 1.0) # Tránh giá trị 0 tuyệt đối
        
        # Resize depth map về kích thước target (ví dụ: 384x384) dùng cv2
        depth = cv2.resize(depth, self.target_size, interpolation=cv2.INTER_LINEAR)
        depth = torch.tensor(depth).unsqueeze(0)  # Thêm chiều kênh
        
        return image, depth

# Thiết lập dữ liệu
image_dir = "./datasets_nyu/rgb_images"
depth_dir = "./datasets_nyu/depth_maps"
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])
dataset = DepthDataset(image_dir, depth_dir, transform, target_size=(384,384))

import matplotlib.pyplot as plt
import random

def show_sample(dataset):
    idx = random.randint(0, len(dataset) - 1)  # Chọn một mẫu ngẫu nhiên
    image, depth = dataset[idx]
    
    # Chuyển tensor sang numpy
    image_np = image.permute(1, 2, 0).numpy()  # Chuyển từ CxHxW sang HxWxC
    depth_np = depth.squeeze(0).numpy()  # Bỏ chiều kênh

    print(image_np.shape)
    print(depth_np.shape)
    
    for depth in depth_np:
        print(depth)
    
    # Chuyển đổi ảnh RGB sang BGR cho OpenCV
    image_bgr = (image_np * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
    
    
    # Hiển thị ảnh với OpenCV
    cv2.imshow("RGB Image", image_bgr)
    cv2.imshow("Depth Map", depth_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Gọi hàm hiển thị
show_sample(dataset)