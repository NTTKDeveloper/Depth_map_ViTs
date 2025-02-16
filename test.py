import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

#Model Depth Map ViTs
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
    depth_dir = "./datasets_nyu/depth_maps"

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
    
    # Đường dẫn tới ảnh đầu vào (thay "input.jpg" bằng ảnh của bạn)
    image_path = "./datasets_nyu/rgb_images/1460.png"
    depth_path = "./datasets_nyu/depth_maps/1460.png"
    
    # Load ảnh bằng OpenCV (ảnh được load dưới dạng BGR)
    img_cv = cv2.imread(image_path)
    depth_sample = cv2.imread(depth_path)
    if img_cv is None:
        raise ValueError("Không tìm thấy ảnh tại đường dẫn: " + image_path)
    
    if depth_sample is None:
        raise ValueError("Không tìm thấy ảnh tại đường dẫn: " + depth_path)
    
    # Resize ảnh về 384x384
    img_cv_resized = cv2.resize(img_cv, (384, 384))
    depth_map_cv_resized = cv2.resize(depth_sample, (384, 384))
    
    # Hiển thị ảnh input
    cv2.imshow("Input Image", img_cv_resized)
    cv2.imshow("Sample Depth", depth_map_cv_resized)
    cv2.waitKey(1)
    
    # Chuyển ảnh từ BGR sang RGB và sang PIL Image
    img_rgb = cv2.cvtColor(img_cv_resized, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Định nghĩa transform: chuyển ảnh thành tensor (giá trị [0,1])
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
    # Ví dụ mong đợi: torch.Size([1, 1, 384, 384])
    
    # Chuyển depth map về numpy, loại bỏ batch và channel nếu cần
    depth_map = predicted_depth.squeeze().cpu().numpy()
    print("Depth Map Min:", depth_map.min(), "Max:", depth_map.max())
    print("Depth Map Shape:", depth_map.shape)
    
    # Chuẩn hóa depth map về khoảng 0-255 để hiển thị (uint8)
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = np.uint8(depth_norm)
    
    # Áp dụng colormap để trực quan hóa depth map
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_BONE)

    end_time = time.time()
    print("Inference time: {:.3f} seconds".format(end_time - start_time))

    # Hiển thị depth map dự đoán
    cv2.imshow("Predicted Depth Map", depth_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
