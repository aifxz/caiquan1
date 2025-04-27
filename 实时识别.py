import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import joblib
import mediapipe as mp

from 数据处理 import get_device  # 你自己写的设备检测
from xgcnn import GestureCNN    # 你的 CNN 模型定义


# ------------ 手部检测初始化 ------------
def init_hand_detection():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )
    return hands, mp.solutions.drawing_utils


# ------------ 手部裁剪函数（增强版）------------
def crop_hand_region(pil_img, hands):
    """
    检测手部区域并裁剪，只返回手部图像（PIL 格式），如果检测失败返回原图。
    """
    img_np = np.array(pil_img)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w, _ = img_np.shape

    results = hands.process(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        x_min, y_min, x_max, y_max = w, h, 0, 0
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

        padding = 30
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        if x_max > x_min and y_max > y_min:
            hand_crop = img_np[y_min:y_max, x_min:x_max]
            return Image.fromarray(hand_crop)

    return pil_img


# ------------ 图像预处理 pipeline ------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),                     # 输入尺寸
    transforms.ToTensor(),                             # 转为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    # ImageNet 均值
                         std=[0.229, 0.224, 0.225])
])

# ------------ 初始化模型和设备 ------------
device = get_device()
print(f"Using device: {device}")

hands, mp_drawing = init_hand_detection()

cnn_model = GestureCNN().to(device)
cnn_model.load_state_dict(torch.load("resnet_feature_extractor.pth", map_location=device))
cnn_model.eval()

xgb_model = joblib.load("xgb_model.pkl")

class_names = ["rock", "paper", "scissors"]

# ------------ 打开摄像头 ------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    # 手部裁剪
    hand_img = crop_hand_region(pil_img, hands)

    # 调试：显示裁剪图像
    cv2.imshow("Cropped Hand", np.array(hand_img))

    # 如果没检测到手（裁剪后图像大小等于原图），就跳过
    if hand_img.size == pil_img.size:
        cv2.putText(frame, "No hand detected", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.imshow("Real-time Rock-Paper-Scissors", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # 图像预处理
    img_tensor = transform(hand_img).unsqueeze(0).to(device)

    # CNN 特征提取
    with torch.no_grad():
        features = cnn_model(img_tensor).cpu().numpy()

    # XGBoost 分类
    pred = xgb_model.predict(features)
    label = class_names[int(pred[0])]

    # 显示结果
    cv2.putText(frame, f"Prediction: {label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Real-time Rock-Paper-Scissors", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
