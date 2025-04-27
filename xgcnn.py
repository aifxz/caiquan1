import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import joblib
import os

from 数据处理 import get_device, init_hand_detection, load_data, crop_hand_region


# 设置固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 定义CNN模型
class GestureCNN(nn.Module):
    def __init__(self):
        super(GestureCNN, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # 新写法
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)

    def forward(self, x):
        return self.resnet(x)


# 提取CNN特征
def extract_cnn_features(loader, cnn_model, device):
    features, labels = [], []
    with torch.no_grad():
        for inputs, label in loader:
            inputs = inputs.to(device)
            output = cnn_model(inputs)
            features.append(output.cpu().numpy())
            labels.append(label.numpy())
    return np.concatenate(features), np.concatenate(labels)


# 训练CNN模型
def train_cnn(train_loader, val_loader, epochs=10, device="cuda"):
    model = GestureCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}, Accuracy: {correct/total}")

    torch.save(model.state_dict(), "resnet_feature_extractor.pth")  # ✅ 保存特征提取器参数


# 训练XGBoost
def train_xgboost(train_loader, val_loader, cnn_model, device, class_names):
    X_train, y_train = extract_cnn_features(train_loader, cnn_model, device)
    X_val, y_val = extract_cnn_features(val_loader, cnn_model, device)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(class_names),
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100
    )
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred)}")

    joblib.dump(xgb_model, "xgb_model.pkl")  # ✅ 保存XGBoost模型
    return xgb_model


# 主程序
if __name__ == "__main__":
    set_seed(42)  # ✅ 设置种子
    device = get_device()
    hands, mp_drawing = init_hand_detection()

    data_dir = r"E:\machinelearning\datacollection\caiquan"
    train_loader, val_loader, class_names = load_data(data_dir, hands, batch_size=32)

    train_cnn(train_loader, val_loader, epochs=10, device=device)

    cnn_model = GestureCNN().to(device)
    cnn_model.load_state_dict(torch.load("resnet_feature_extractor.pth"))
    cnn_model.eval()

    xgb_model = train_xgboost(train_loader, val_loader, cnn_model, device, class_names)
