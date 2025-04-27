# inference.py
import torch
import cv2
from 推理代码 import FineTuneResNet
from 数据处理 import get_device, crop_hand_region, init_hand_detection
from torchvision import transforms
import os

if __name__ == "__main__":
    # 获取设备
    device = get_device()

    # 类别名字手动指定一下（要跟训练时一致）
    class_names = ["rock", "paper", "scissors"]  # 这里根据你的实际类别填！

    # 初始化手部检测器
    hands, mp_drawing = init_hand_detection()

    # 初始化模型并加载训练好的权重
    model = FineTuneResNet(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load("model/layer4/epoch5.pth"),strict=Falseqq)  # 加载最后保存的模型
    model.eval()

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 使用 MediaPipe 手部检测
        image_pil = crop_hand_region(frame, hands, save_cropped=False)

        # 转换成Tensor
        image_tensor = transforms.ToTensor()(image_pil).unsqueeze(0).to(device)

        # 进行预测
        with torch.no_grad():
            output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        prediction = class_names[predicted.item()]
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()

        # 在画面上显示
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"{prediction} ({confidence:.2f})", (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Real-time Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
