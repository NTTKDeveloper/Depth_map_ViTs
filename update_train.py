import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from PIL import Image
from model import DepthVisionTransformer  # Giả sử mô hình của bạn được định nghĩa trong file này

# -----------------------------------------------------
# Định nghĩa SILogLoss
# -----------------------------------------------------
class SILogLoss(nn.Module):
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None):
        if mask is not None:
            input = input[mask]
            target = target[mask]
        input = torch.clamp(input, min=1e-6)
        target = torch.clamp(target, min=1e-6)
        g = torch.log(input) - torch.log(target)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)

# -----------------------------------------------------
# Định nghĩa Dataset với tùy chọn cache toàn bộ dữ liệu vào RAM
# -----------------------------------------------------
class DepthDataset(Dataset):
    def __init__(self, image_dir, depth_dir, transform=None, target_size=(384,384), cache_data=False):
        """
        image_dir: thư mục chứa ảnh RGB
        depth_dir: thư mục chứa depth map
        transform: các phép biến đổi (ví dụ Resize, ToTensor)
        target_size: kích thước mong muốn của ảnh và depth map
        cache_data: nếu True, toàn bộ dữ liệu sẽ được đọc và lưu vào RAM khi khởi tạo
        """
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.depth_filenames = sorted(os.listdir(depth_dir))
        self.transform = transform
        self.target_size = target_size
        self.cache_data = cache_data

        if self.cache_data:
            print("Caching dataset into RAM...")
            self.cached_data = []
            for i in range(len(self.image_filenames)):
                # Đọc ảnh RGB
                image_path = os.path.join(self.image_dir, self.image_filenames[i])
                image = Image.open(image_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                
                # Đọc và xử lý depth map
                depth_path = os.path.join(self.depth_dir, self.depth_filenames[i])
                depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                depth = cv2.resize(depth, self.target_size, interpolation=cv2.INTER_LINEAR)
                depth = depth / 255.0           # Chuẩn hóa về khoảng 0-1
                depth = np.clip(depth, 1e-6, 1.0) # Tránh giá trị 0 tuyệt đối
                depth = torch.tensor(depth).unsqueeze(0)  # Thêm kênh cho depth map
                self.cached_data.append((image, depth))
            print(f"Đã cache {len(self.cached_data)} samples.")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if self.cache_data:
            return self.cached_data[idx]
        else:
            # Nếu không cache, thực hiện quy trình đọc và xử lý dữ liệu như trên
            image_path = os.path.join(self.image_dir, self.image_filenames[idx])
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            
            depth_path = os.path.join(self.depth_dir, self.depth_filenames[idx])
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            depth = depth / 255.0
            depth = np.clip(depth, 1e-6, 1.0)
            depth = cv2.resize(depth, self.target_size, interpolation=cv2.INTER_LINEAR)
            depth = torch.tensor(depth).unsqueeze(0)
            return image, depth

# -----------------------------------------------------
# Hàm main chứa toàn bộ quy trình khởi tạo dữ liệu, mô hình và huấn luyện
# -----------------------------------------------------
def main():
    # Thiết lập đường dẫn và biến đổi dữ liệu
    image_dir = "./datasets_nyu/rgb_images"
    depth_dir = "./datasets_nyu/depth_maps"
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])
    
    # Cache toàn bộ dataset vào RAM
    dataset = DepthDataset(image_dir, depth_dir, transform, target_size=(384,384), cache_data=True)
    
    # Với dữ liệu đã cache, có thể sử dụng num_workers=0 để đảm bảo DataLoader trả về dữ liệu từ RAM
    dataloader = DataLoader(dataset, batch_size=7, shuffle=True, num_workers=0, pin_memory=True)
    
    # Khởi tạo hoặc load mô hình
    model_path = "model/depth_model.pth"
    os.makedirs("model", exist_ok=True)
    if os.path.exists(model_path):
        model = torch.load(model_path)
        print("Loaded existing model.")
    else:
        model = DepthVisionTransformer()
        print("No existing model found. Initializing new model.")
    
    criterion = SILogLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model.train()
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (images, depths) in enumerate(dataloader):
            # Chuyển batch hiện tại từ RAM sang VRAM
            images = images.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.clamp(outputs, min=1e-6)
            loss = criterion(outputs, depths)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
            
            # Sau khi xử lý batch hiện tại, xóa các biến trên GPU để giải phóng VRAM
            del images, depths, outputs
            torch.cuda.empty_cache()  # (Tùy chọn: chỉ dùng nếu cần ép giải phóng bộ nhớ ngay lập tức)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss/len(dataloader):.4f}")
        torch.save(model, model_path)
        print("Huấn luyện hoàn tất một epochs! Model saved.")
    
    torch.save(model, model_path)
    print("Huấn luyện hoàn tất! Model saved.")

# -----------------------------------------------------
# Bảo vệ điểm vào khi chạy trên Windows
# -----------------------------------------------------
if __name__ == '__main__':
    main()
