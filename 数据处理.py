import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# 检查 CUDA 是否可用
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

# 初始化 MediaPipe 手部检测
def init_hand_detection():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils
    return hands, mp_drawing

# 手部裁剪函数
def crop_hand_region(image, hands, save_cropped=False):
    # 转换为OpenCV格式
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = hands.process(cv_img)

    if results.multi_hand_landmarks:
        h, w, _ = cv_img.shape
        x_coords = [lm.x * w for lm in results.multi_hand_landmarks[0].landmark]
        y_coords = [lm.y * h for lm in results.multi_hand_landmarks[0].landmark]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # 扩展框防止裁得太紧
        margin = 30  # 扩大裁剪区域的边距
        x_min = max(x_min - margin, 0)
        x_max = min(x_max + margin, w)
        y_min = max(y_min - margin, 0)
        y_max = min(y_max + margin, h)

        hand_crop = cv_img[y_min:y_max, x_min:x_max]
        hand_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)

        if save_cropped:
            cropped_image = Image.fromarray(hand_crop)
            cropped_image.save("cropped_image.jpg")  # 保存裁剪后的图像

        return Image.fromarray(hand_crop)
    else:
        return image  # 如果没检测到手，就返回原图

# 数据预处理与加载函数

def load_data(data_dir, hands, batch_size=32, val_ratio=0.2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class HandCroppedDataset(datasets.ImageFolder):
        def __getitem__(self, index):
            path, target = self.samples[index]
            image = self.loader(path)

            if hands is not None:
                image = crop_hand_region(image, hands, save_cropped=False)

            if self.transform is not None:
                image = self.transform(image)
            return image, target

    dataset = HandCroppedDataset(root=data_dir, transform=transform)

    # 划分训练集和验证集
    train_size = int((1 - val_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dataset.classes

# 测试代码：预览部分裁剪结果
def preview_images(train_loader, class_names):
    import matplotlib.pyplot as plt
    import numpy as np

    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    hands, _ = init_hand_detection()

    fig, axes = plt.subplots(1, 5, figsize=(10, 5))
    for i in range(5):
        ax = axes[i]
        img = images[i].numpy().transpose((1, 2, 0))  # 转换为 HWC 格式
        ax.imshow(img)
        ax.set_title(f"Label: {class_names[labels[i].item()]}")  # 显示标签
        ax.axis('off')
    plt.show()

# 主程序入口
if __name__ == "__main__":
    data_dir = r"E:\machinelearning\datacollection\caiquan"  # 数据集路径

    # 获取设备和初始化手部检测
    device = get_device()
    hands, mp_drawing = init_hand_detection()

    # 加载数据
    train_loader, val_loader, class_names = load_data(data_dir, hands)

    # 查看某个批次图像
    preview_images(train_loader, class_names)
