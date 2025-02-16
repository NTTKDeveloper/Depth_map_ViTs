import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================================================
# 1. Lớp PatchEmbed: Tách ảnh thành các patch và chiếu sang không gian đặc trưng
# ========================================================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_chans=3, embed_dim=768):
        """
        img_size: kích thước ảnh đầu vào (384x384)
        patch_size: kích thước patch (16x16)
        in_chans: số kênh của ảnh (3 cho RGB)
        embed_dim: số chiều vector embedding sau khi chiếu patch
        """
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # Số lượng patch: (384/16) * (384/16) = 24*24 = 576
        self.num_patches = (img_size // patch_size) ** 2
        # Dùng Conv2d với kernel = patch_size và stride = patch_size để tách patch
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)              # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

# ========================================================================
# 2. Lớp MaskedMultiheadAttention: Multi-Head Attention có hỗ trợ mask (nếu cần)
# ========================================================================
class MaskedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        """
        embed_dim: số chiều của embedding (768)
        num_heads: số đầu attention (ví dụ 12)
        dropout: dropout cho attention
        """
        super(MaskedMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim phải chia hết cho num_heads"
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, N, 3*embed_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # mỗi tensor: [B, num_heads, N, head_dim]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = attn @ v  # [B, num_heads, N, head_dim]
        x = x.transpose(1, 2).reshape(B, N, C)  # [B, N, embed_dim]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ========================================================================
# 3. Lớp MLP: MLP đơn giản cho transformer block
# ========================================================================
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        """
        in_features: số chiều của input
        hidden_features: số chiều của lớp ẩn, mặc định bằng in_features nếu không truyền vào
        out_features: số chiều của output, mặc định bằng in_features nếu không truyền vào
        dropout: tỷ lệ dropout
        """
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ========================================================================
# 4. Transformer Encoder Block
# ========================================================================
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, attention_dropout=0.0):
        """
        embed_dim: số chiều embedding
        num_heads: số đầu attention
        mlp_ratio: tỉ số chiều của lớp ẩn so với embed_dim
        dropout: dropout chung
        attention_dropout: dropout trong attention
        """
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MaskedMultiheadAttention(embed_dim, num_heads, dropout=attention_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_dim, dropout=dropout)
        
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x

# ========================================================================
# 5. Mô hình Depth Vision Transformer: Dự đoán Depth Map
# ========================================================================
class DepthVisionTransformer(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 dropout=0.0, attention_dropout=0.0):
        """
        img_size: kích thước ảnh đầu vào (384)
        patch_size: kích thước patch (16)
        in_chans: số kênh của ảnh (3 cho RGB)
        embed_dim: số chiều embedding (768)
        depth: số transformer block (ví dụ: 12)
        num_heads: số đầu attention (ví dụ: 12)
        mlp_ratio: tỉ số chiều của MLP so với embed_dim
        dropout: dropout chung
        attention_dropout: dropout trong attention
        """
        super(DepthVisionTransformer, self).__init__()
        # Lưu embed_dim vào thuộc tính để dùng trong forward
        self.embed_dim = embed_dim

        # Patch embedding: tách ảnh thành các patch và chiếu thành vector embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches  # ví dụ: 576
        
        # Positional embedding cho các patch (không dùng token [cls])
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout, attention_dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Decoder: chuyển đổi đặc trưng từ không gian patch thành depth map.
        # Sau khi patch embedding, ảnh có kích thước 24x24 (384/16=24).
        # Sử dụng upsample với scale_factor=16 để đạt kích thước 384x384.
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dim, 1, kernel_size=1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_module)
    
    def _init_module(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x, attn_mask=None):
        B = x.shape[0]
        # 1. Patch embedding: [B, num_patches, embed_dim]
        x = self.patch_embed(x)
        # 2. Thêm positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 3. Transformer encoder blocks
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.norm(x)  # [B, num_patches, embed_dim]
        
        # 4. Reshape chuỗi token thành tensor không gian: [B, embed_dim, H_patch, W_patch]
        H_patch = W_patch = int(self.num_patches ** 0.5)  # ví dụ: 24
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H_patch, W_patch)
        
        # 5. Dự đoán depth map: output shape [B, 1, 384, 384]
        # depth_map = self.decoder(x)
        # depth_map = torch.sigmoid(self.decoder(x))
        depth_map = F.softplus(self.decoder(x))
        return depth_map

# # ========================================================================
# # Main: Load ảnh, chuẩn bị input và hiển thị kết quả depth map
# # ========================================================================
# if __name__ == "__main__":
#     # Thiết lập thiết bị: GPU nếu có, ngược lại CPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Khởi tạo mô hình Depth Vision Transformer và đưa lên thiết bị
#     model = DepthVisionTransformer().to(device)
#     model.eval()  # Chế độ inference
    
#     # Đường dẫn tới ảnh đầu vào (thay "input.jpg" bằng ảnh của bạn)
#     image_path = "./test_imgs/1.jpg"
    
#     # Load ảnh bằng OpenCV (ảnh được load dưới dạng BGR)
#     img_cv = cv2.imread(image_path)
#     if img_cv is None:
#         raise ValueError("Không tìm thấy ảnh tại đường dẫn: " + image_path)
    
#     # Resize ảnh về 384x384
#     img_cv_resized = cv2.resize(img_cv, (384, 384))
    
#     # Hiển thị ảnh input
#     cv2.imshow("Input Image", img_cv_resized)
#     cv2.waitKey(1)
    
#     # Chuyển ảnh từ BGR sang RGB và sang PIL Image
#     img_rgb = cv2.cvtColor(img_cv_resized, cv2.COLOR_BGR2RGB)
#     img_pil = Image.fromarray(img_rgb)
    
#     # Định nghĩa transform: chuyển ảnh thành tensor (giá trị [0,1])
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#     # Thêm batch dimension: [1, 3, 384, 384]
#     img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
#     # Bắt đầu đo thời gian
#     start_time = time.time()

#     # Dự đoán depth map
#     with torch.no_grad():
#         predicted_depth = model(img_tensor)
    
#     print("Predicted Depth Tensor Shape:", predicted_depth.shape)
#     # Ví dụ mong đợi: torch.Size([1, 1, 384, 384])
    
#     # Chuyển depth map về numpy, loại bỏ batch và channel nếu cần
#     depth_map = predicted_depth.squeeze().cpu().numpy()
#     print("Depth Map Min:", depth_map.min(), "Max:", depth_map.max())
#     print("Depth Map Shape:", depth_map.shape)
    
#     # Chuẩn hóa depth map về khoảng 0-255 để hiển thị (uint8)
#     depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
#     depth_norm = np.uint8(depth_norm)
    
#     # Áp dụng colormap để trực quan hóa depth map
#     depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

#     end_time = time.time()
#     print("Inference time: {:.3f} seconds".format(end_time - start_time))

#     # Hiển thị depth map dự đoán
#     cv2.imshow("Predicted Depth Map", depth_color)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
