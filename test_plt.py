import os
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# Model Depth Map ViTs
from model import DepthVisionTransformer

# ========================================================================
# Main: Load ảnh, chuẩn bị input và hiển thị kết quả depth map
# ========================================================================
if __name__ == "__main__":
    # Thiết lập thiết bị: GPU nếu có, ngược lại CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load mô hình nếu có, nếu không khởi tạo mới
    model_path = "model/depth_model.pth"
    os.makedirs("model", exist_ok=True)

    model = DepthVisionTransformer()
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded existing model.")
    else:
        print("No existing model found. Initializing new model.")
        exit()

    # Đưa mô hình lên thiết bị và chuyển sang chế độ inference
    model.to(device)
    model.eval()
    
    # Đường dẫn tới ảnh đầu vào
    # n = 501
    # image_path = f"./datasets_SUN-RGBD 2D/rgb_images/{n}.jpg"
    # depth_path = f"./datasets_SUN-RGBD 2D/depth_maps/{n}.png"

    n = 1
    image_path = f"./datasets_nyu/rgb_images/{n}.png"
    depth_path = f"./datasets_nyu/depth_maps/{n}.png"
    
    # Load ảnh bằng OpenCV
    img_cv = cv2.imread(image_path)
    depth_sample = cv2.imread(depth_path)
    if img_cv is None:
        raise ValueError("Không tìm thấy ảnh tại đường dẫn: " + image_path)
    
    if depth_sample is None:
        raise ValueError("Không tìm thấy ảnh tại đường dẫn: " + depth_path)
    
    # Resize ảnh về 384x384
    img_cv_resized = cv2.resize(img_cv, (384, 384))
    depth_map_cv_resized = cv2.resize(depth_sample, (384, 384))
    
    # Chuyển ảnh từ BGR sang RGB và sang PIL Image
    img_rgb = cv2.cvtColor(img_cv_resized, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Định nghĩa transform: chuyển ảnh thành tensor và chuẩn hóa
    transform = transforms.Compose([
        transforms.ToTensor(),  # Chuyển ảnh thành tensor có giá trị [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa
    ])
    
    # Chuyển đổi ảnh thành tensor và đưa lên thiết bị
    img_tensor = transform(img_pil).unsqueeze(0).to(device)  # Thêm batch dimension

    # Bắt đầu đo thời gian inference
    start_time = time.time()

    # Dự đoán depth map
    with torch.no_grad():
        predicted_depth = model(img_tensor)
    
    end_time = time.time()
    print("Inference time: {:.3f} seconds".format(end_time - start_time))
    print("Predicted Depth Tensor Shape:", predicted_depth.shape)
    
    # Chuyển depth map về numpy
    depth_map = predicted_depth.squeeze().cpu().numpy()
    print("Depth Map Min:", depth_map.min(), "Max:", depth_map.max())
    
    # Chuẩn hóa depth map về khoảng 0-255 để hiển thị
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = np.uint8(depth_norm)
    
    # Áp dụng colormap để trực quan hóa depth map
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_BONE)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)  # Chuyển về RGB

    # Giải chuẩn hóa ảnh đầu vào để hiển thị (nếu cần)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_denormalized = img_tensor.squeeze(0).cpu() * std + mean  # Giải chuẩn hóa
    img_denormalized = img_denormalized.permute(1, 2, 0).numpy()  # Đưa về HWC để hiển thị
    
    # Hiển thị ảnh đầu vào và depth map sử dụng matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_denormalized)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    axes[1].imshow(cv2.cvtColor(depth_map_cv_resized, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Sample Depth")
    axes[1].axis("off")
    
    axes[2].imshow(depth_color)
    axes[2].set_title("Predicted Depth Map")
    axes[2].axis("off")
    
    plt.show()
