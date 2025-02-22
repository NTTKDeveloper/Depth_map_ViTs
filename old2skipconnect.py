import torch
import torch.nn as nn
import torch.nn.functional as F

# THÊM MỚI: Import mô hình cũ và mô hình mới từ các file riêng biệt
from model import DepthVisionTransformer as OldDepthVisionTransformer
from model_skip_connect import DepthVisionTransformer as NewDepthVisionTransformer


# ========================================================================
# THÊM MỚI: Hàm chuyển thông số từ mô hình cũ sang mô hình mới
# ========================================================================
def transfer_parameters(old_checkpoint_path, new_model):
    """
    Hàm chuyển thông số từ mô hình cũ sang mô hình mới.
    old_checkpoint_path: đường dẫn file checkpoint của mô hình cũ (model.py)
    new_model: instance của mô hình mới (model_skip_connect.py)
    """
    # Load checkpoint của mô hình cũ
    old_state_dict = torch.load(old_checkpoint_path, map_location='cpu')
    
    # THÊM MỚI: Sử dụng strict=False để bỏ qua những key không khớp do khác biệt cấu trúc (như skip connection)
    missing_keys, unexpected_keys = new_model.load_state_dict(old_state_dict, strict=False)
    
    if missing_keys:
        print("Missing keys:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys:", unexpected_keys)
    print("Chuyển thông số thành công!")
    return new_model

if __name__ == "__main__":
    # THÊM MỚI: Đường dẫn tới checkpoint của mô hình cũ (đã được lưu từ model.py)
    old_checkpoint_path = "./model/old_depth_model.pth"
    
    # THÊM MỚI: Khởi tạo mô hình mới (được định nghĩa trong model_skip_connect.py)
    new_model = NewDepthVisionTransformer()
    
    # THÊM MỚI: Chuyển thông số từ mô hình cũ sang mô hình mới
    new_model = transfer_parameters(old_checkpoint_path, new_model)
    
    # THÊM MỚI: Lưu mô hình mới với thông số đã chuyển đổi
    torch.save(new_model.state_dict(), "./model/new_model_with_skip.pth")
    
    print("Quá trình chuyển đổi thông số hoàn tất!")
