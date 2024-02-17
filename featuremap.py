from cv2 import threshold
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from models.backbone.resnet import resnet50
import torch.nn as nn
import torch.nn.functional as F

class RPN(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RPN, self).__init__()

        self.num_anchors = num_anchors

        # Intermediate convolutional layer
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1)

        # Classification layer for anchor scores
        self.cls_layer = nn.Conv2d(512, num_anchors, kernel_size=1, stride=1)

        # Regression layer for bounding box offsets
        self.reg_layer = nn.Conv2d(512, num_anchors * 4, kernel_size=1, stride=1)

    def forward(self, x):
        # Intermediate feature representation
        x = F.relu(self.conv(x))

        # Anchor scores
        cls_scores = self.cls_layer(x)

        # Bounding box offsets
        reg_offsets = self.reg_layer(x)

        # Reshape the output to have the same number of anchors for each spatial position
        cls_scores = cls_scores.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_anchors)
        reg_offsets = reg_offsets.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_anchors * 4)

        return cls_scores, reg_offsets

# Bước 2: Tạo một instance của mô hình ResNet-50
model = resnet50(pretrained=True, in_channels=3)

# Bước 3: Load ảnh và tiền xử lý
image_path = r"C:\Users\OS\Desktop\textDetection\MyTask\002.jpg"
image = Image.open(image_path)

# Áp dụng các biến đổi cần thiết
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # Thêm chiều batch

# Bước 5: Chạy ảnh qua mô hình để nhận được đầu ra
with torch.no_grad():
    model.eval()
    output = model(input_batch)

# Forward pass to get RPN predictions
feature_maps = output[-1].squeeze().cpu().numpy()
print(feature_maps.shape[0])
print('len', len(feature_maps))
in_channels_resnet=feature_maps.shape[0]
num_anchors = 9
rpn = RPN(in_channels_resnet, num_anchors)
rpn_cls_scores, rpn_reg_offsets = rpn(output[-1])

# Hiển thị kích thước của các tensor đầu ra
print("RPN Classification Scores Shape:", rpn_cls_scores.shape)
print("RPN Regression Offsets Shape:", rpn_reg_offsets.shape)

import numpy as np

# Hàm chuyển đổi anchor boxes thành bounding boxes
def generate_boxes(cls_scores, reg_offsets, anchors, threshold=0.7):
    # Sử dụng detach để không ghi lại gradient khi chuyển đổi sang NumPy
    cls_scores = cls_scores.numpy()
    reg_offsets = reg_offsets.numpy()
    anchors = anchors
    
    # Số lượng anchors
    num_anchors = cls_scores.shape[2]

    # Số điểm trên feature map
    num_points = cls_scores.shape[1]

    # Khởi tạo danh sách để lưu trữ bounding boxes
    boxes = []

    # Duyệt qua từng điểm trên feature map
    for i in range(num_points):
        for j in range(num_points):
            # Duyệt qua từng anchor box
            for k in range(num_anchors):
                # Lấy điểm số phân loại cho anchor box này
                score = cls_scores[0, i, j, k]

                # Nếu điểm số lớn hơn một ngưỡng, xem xét bounding box
                if score > threshold:
                    # Lấy offset cho anchor box này
                    offset = reg_offsets[0, i, j, k * 4:(k + 1) * 4]

                    # Lấy anchor box tương ứng
                    anchor = anchors[0, i, j, k * 4:(k + 1) * 4]

                    # Áp dụng offset để có bounding box dự đoán
                    box = apply_offset(anchor, offset)

                    # Thêm bounding box và điểm số phân loại vào danh sách
                    boxes.append((box, score))

    return boxes

# Hàm áp dụng offset để có bounding box dự đoán
def apply_offset(anchor, offset):
    # anchor: [x_center, y_center, width, height]
    # offset: [delta_x_center, delta_y_center, delta_width, delta_height]

    # Tính toán giá trị mới cho bounding box
    x_center = anchor[0] + offset[0] * anchor[2]
    y_center = anchor[1] + offset[1] * anchor[3]
    width = anchor[2] * np.exp(offset[2])
    height = anchor[3] * np.exp(offset[3])

    # Chuyển đổi về dạng [x_min, y_min, x_max, y_max]
    x_min = x_center - 0.5 * width
    y_min = y_center - 0.5 * height
    x_max = x_center + 0.5 * width
    y_max = y_center + 0.5 * height

    return [x_min, y_min, x_max, y_max]
def generate_anchors(feature_map_size, num_anchors=9, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
    stride = 1  # Bước nhảy của anchor

    # Tính toán tâm của anchors
    y, x = np.meshgrid(np.arange(0, feature_map_size[0]), np.arange(0, feature_map_size[1]))
    y, x = y * stride, x * stride
    centers = np.vstack([y.ravel(), x.ravel(), y.ravel(), x.ravel()]).transpose()

    # Tính toán kích thước của anchors với các tỉ lệ và tỷ lệ
    anchors = []
    for ratio in ratios:
        for scale in scales:
            h = scale * np.sqrt(ratio)
            w = scale * np.sqrt(1 / ratio)
            anchor = np.array([-w / 2, -h / 2, w / 2, h / 2])
            anchors.append(centers + anchor)

    return np.array(anchors)
# Hàm vẽ bounding boxes lên ảnh
def draw_boxes(image, boxes):
    import matplotlib.patches as patches

    # Hiển thị ảnh
    plt.imshow(image)

    # Tạo một subplot
    ax = plt.gca()

    # Vẽ bounding boxes
    for box in boxes:
        rect = patches.Rectangle(
            (box[0][0], box[0][1]),
            box[0][2] - box[0][0],
            box[0][3] - box[0][1],
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )

        # Thêm bounding box vào subplot
        ax.add_patch(rect)

    # Hiển thị ảnh với bounding boxes
    plt.show()
feature_map_size = (7, 7)
anchors = generate_anchors(feature_map_size)
# Chọn một điểm trên feature map và chuyển đổi các anchor boxes thành bounding boxes
point_index = 8 # Chọn một điểm trên feature map (điều chỉnh nếu cần)
boxes = generate_boxes(rpn_cls_scores, rpn_reg_offsets, anchors[point_index])

# Hiển thị bounding boxes trên ảnh gốc
draw_boxes(np.array(image), boxes)

