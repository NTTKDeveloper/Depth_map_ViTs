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
from tqdm import tqdm  # Import tqdm để tạo thanh tiến trình

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
# Định nghĩa SSIMLoss
# -----------------------------------------------------
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = None

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2 * sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, sigma=1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        # Giả sử đầu vào có kích thước (N, C, H, W)
        (_, channel, _, _) = img1.size()
        if self.window is None or self.window.size(0) != channel:
            self.window = self.create_window(self.window_size, channel).to(img1.device)
        # Tính trung bình cục bộ
        mu1 = nn.functional.conv2d(img1, self.window, padding=self.window_size//2, groups=channel)
        mu2 = nn.functional.conv2d(img2, self.window, padding=self.window_size//2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Tính phương sai cục bộ
        sigma1_sq = nn.functional.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = nn.functional.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=channel) - mu1_mu2

        # Các hằng số ổn định
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

# -----------------------------------------------------
# Định nghĩa CombinedLoss kết hợp SILogLoss và SSIMLoss
# -----------------------------------------------------
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        alpha: trọng số cho SILogLoss. (1 - alpha) sẽ là trọng số cho SSIM loss.
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.silog = SILogLoss()
        self.ssim = SSIMLoss()

    def forward(self, input, target):
        # Tính SILog loss
        loss_silog = self.silog(input, target)
        # Tính SSIM loss; vì SSIM đo độ tương đồng, nên loss được định nghĩa là 1 - SSIM.
        loss_ssim = 1 - self.ssim(input, target)
        # Kết hợp loss theo trọng số
        return self.alpha * loss_silog + (1 - self.alpha) * loss_ssim, loss_silog, loss_ssim

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
                depth = depth / 255.0           # Chuẩn hóa về khoảng [0, 1]
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
    train_image_dir = "./datasets_nyu/rgb_images"
    train_depth_dir = "./datasets_nyu/depth_maps"
    
    val_image_dir = "./datasets_SUN-RGBD 2D/rgb_images"
    val_depth_dir = "./datasets_SUN-RGBD 2D/depth_maps"

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])

    # Dataset train (cache vào RAM)
    train_dataset = DepthDataset(train_image_dir, train_depth_dir, transform, target_size=(384, 384), cache_data=True)

    # Dataset validation (KHÔNG cache vào RAM để tiết kiệm bộ nhớ)
    val_dataset = DepthDataset(val_image_dir, val_depth_dir, transform, target_size=(384, 384), cache_data=True)

    # DataLoader cho tập train
    train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=0, pin_memory=True)

    # DataLoader cho tập validation (không cần shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=3, shuffle=False, num_workers=0, pin_memory=True)

    # Khởi tạo hoặc load mô hình
    model_path = "model/depth_model.pth"
    os.makedirs("model", exist_ok=True)
    model = DepthVisionTransformer()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded existing model.")
    else:
        print("No existing model found. Initializing new model.")

    # Đưa mô hình lên GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Sử dụng CombinedLoss thay vì chỉ SILogLoss
    criterion = CombinedLoss(alpha=0.7)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    model.train()
    num_epochs = 1
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_loss_ssim = 0.0
        total_loss_silog = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        # Huấn luyện trên tập train
        for batch_idx, (images, depths) in enumerate(progress_bar):
            images, depths = images.to(device, non_blocking=True), depths.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.clamp(outputs, min=1e-6)
            loss, loss_silog, loss_ssim = criterion(outputs, depths)
            loss.backward() # Tính gradient qua backpropagation, lưu vào .grad của các tham số để cập nhật trọng số.
            optimizer.step() # Cập nhật trọng số
            
            # Tính tổng cộng tất cả các loại loss
            total_loss += loss.item()
            total_loss_ssim += loss_ssim.item()
            total_loss_silog += loss_silog.item()

            del images, depths, outputs
            torch.cuda.empty_cache()

        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_loss_ssim = total_loss_ssim / len(train_dataloader)
        avg_train_loss_silog = total_loss_silog / len(train_dataloader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Loss_ssim: {avg_train_loss_ssim:.4f}, Loss_silog: {avg_train_loss_silog:.4f}")

        # Đánh giá trên tập validation
        model.eval()
        val_loss = 0.0
        val_loss_ssim = 0.0
        val_loss_silog = 0.0
        with torch.no_grad():
            for images, depths in val_dataloader:
                images, depths = images.to(device, non_blocking=True), depths.to(device, non_blocking=True)
                outputs = model(images)
                outputs = torch.clamp(outputs, min=1e-6)
                loss, loss_silog, loss_ssim = criterion(outputs, depths)

                # Tính tổng cộng tất cả các loại loss
                val_loss += loss.item()
                val_loss_ssim += loss_ssim.item()
                val_loss_silog += loss_silog.item()

                del images, depths, outputs
                torch.cuda.empty_cache()

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_loss_ssim = val_loss_ssim / len(val_dataloader)
        avg_val_loss_silog = val_loss_silog / len(val_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Loss_ssim: {avg_val_loss_ssim:.4f}, Loss_silog: {avg_val_loss_silog:.4f}")

        # Lưu mô hình
        torch.save(model.state_dict(), model_path)
        print("Model saved.")

    print("Huấn luyện hoàn tất! Model saved.")

    # Sau khi huấn luyện xong, tự động tắt máy (lệnh cho Windows 11)
    # print("Huấn luyện xong! Hệ thống sẽ tự động tắt trong vài giây...")
    # os.system("shutdown /s /t 0")

# -----------------------------------------------------
# Bảo vệ điểm vào khi chạy trên Windows
# -----------------------------------------------------
if __name__ == '__main__':
    main()
