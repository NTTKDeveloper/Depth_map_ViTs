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
        model.load_state_dict(torch.load(model_path))
        print("Loaded existing model.")
    else:
        print("No existing model found. Initializing new model.")
        exit()

    # Khởi tạo mô hình Depth Vision Transformer và đưa lên thiết bị
    model.to(device)
    model.eval()  # Chế độ inference
    
    # Đường dẫn tới ảnh đầu vào
    n = 2
    image_path = f"./datasets_SUN-RGBD 2D/rgb_images/{n}.jpg"
    depth_path = f"./datasets_SUN-RGBD 2D/depth_maps/{n}.png"
    
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
    
    # Định nghĩa transform: chuyển ảnh thành tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Thêm batch dimension: [1, 3, 384, 384]
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    # Bắt đầu đo thời gian
    start_time = time.time()

    # Dự đoán depth map
    with torch.no_grad():
        predicted_depth = model(img_tensor)
    
    print("Predicted Depth Tensor Shape:", predicted_depth.shape)
    
    # Chuyển depth map về numpy
    depth_map = predicted_depth.squeeze().cpu().numpy()
    print("Depth Map Min:", depth_map.min(), "Max:", depth_map.max())
    print("Depth Map Shape:", depth_map.shape)
    
    # Chuẩn hóa depth map về khoảng 0-255
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = np.uint8(depth_norm)
    
    # Áp dụng colormap để trực quan hóa depth map
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_BONE)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)  # Chuyển về RGB

    end_time = time.time()
    print("Inference time: {:.3f} seconds".format(end_time - start_time))

    # Hiển thị ảnh đầu vào và depth map sử dụng matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    axes[1].imshow(cv2.cvtColor(depth_map_cv_resized, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Sample Depth")
    axes[1].axis("off")
    
    axes[2].imshow(depth_color)
    axes[2].set_title("Predicted Depth Map")
    axes[2].axis("off")
    
    plt.show()
