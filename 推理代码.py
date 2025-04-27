import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import cv2
import numpy as np


# # 定义微调版的ResNet模型
# class FineTuneResNet(nn.Module):
#     def __init__(self, num_classes):
#         super(FineTuneResNet, self).__init__()
#         # 加载预训练的ResNet18模型
#         self.backbone = models.resnet18(pretrained=True)
#
#         # 冻结所有参数，防止一开始就训练整个网络
#         for param in self.backbone.parameters():
#             param.requires_grad = False
#
#         # 替换最后的全连接层，使其适配自己的类别数
#         in_features = self.backbone.fc.in_features
#         self.backbone.fc = nn.Linear(in_features, num_classes)
#
#     def forward(self, x):
#         return self.backbone(x)
#
#
# # 设置训练阶段
# def setup_training_stage(model, stage="fc", base_lr=1e-3):
#     # 先把所有参数冻结
#     for param in model.parameters():
#         param.requires_grad = False
#
#     # 根据训练阶段，解冻特定层
#     if stage == "fc":
#         # 只训练最后的全连接层
#         for param in model.backbone.fc.parameters():
#             param.requires_grad = True
#     elif stage == "layer4":
#         # 解冻layer4和fc层
#         for param in model.backbone.layer4.parameters():
#             param.requires_grad = True
#         for param in model.backbone.fc.parameters():
#             param.requires_grad = True
#     else:
#         raise ValueError("Unknown stage: should be 'fc' or 'layer4'")
#
#     # 只优化需要更新的参数
#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr)
#     return optimizer
#
#
# # 训练函数
# def train_model(model, train_loader, val_loader, optimizer, device, epochs=10, save_path=None):
#     criterion = nn.CrossEntropyLoss()
#
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item() * inputs.size(0)
#             _, preds = torch.max(outputs, 1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#
#         train_loss = running_loss / total
#         train_acc = correct / total
#
#         # 验证
#         val_loss, val_acc = evaluate_model(model, val_loader, device, criterion)
#
#         print(f"Epoch {epoch + 1}/{epochs}: "
#               f"Train Loss={train_loss:.4f}  Train Acc={train_acc:.4f}  "
#               f"Val Loss={val_loss:.4f}  Val Acc={val_acc:.4f}")
#
#     # 保存模型
#     if save_path:
#         torch.save(model.state_dict(), save_path + ".pth")
#         print(f"✅ 模型保存到 {save_path}.pth")
#
#
# # 验证函数
# def evaluate_model(model, val_loader, device, criterion):
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#
#             running_loss += loss.item() * inputs.size(0)
#             _, preds = torch.max(outputs, 1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#
#     val_loss = running_loss / total
#     val_acc = correct / total
#
#     return val_loss, val_acc
# ✅ 模型定义
class FineTuneResNet(nn.Module):
    def __init__(self, num_classes):
        super(FineTuneResNet, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# ✅ 训练阶段设置
def setup_training_stage(model, stage="fc", base_lr=1e-3):
    for param in model.parameters():
        param.requires_grad = False

    if stage == "fc":
        for param in model.resnet.fc.parameters():
            param.requires_grad = True
    elif stage == "layer4":
        for name, param in model.resnet.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
    else:
        raise ValueError("stage 参数必须为 'fc' 或 'layer4'")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr
    )
    return optimizer

# ✅ 模型训练
def train_model(model, train_loader, val_loader, optimizer, device, epochs=5, save_path="model"):
    criterion = nn.CrossEntropyLoss()

    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Acc: {acc:.4f}")

        # ✅ 保存中间模型
        torch.save(model.state_dict(), os.path.join(save_path, f"epoch{epoch+1}.pth"))

        # ✅ 每轮评估一次
        evaluate_model(model, val_loader, device, show_matrix=True)

# ✅ 模型评估 + 混淆矩阵
def evaluate_model(model, dataloader, device, show_matrix=False, class_names=None):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    print(f"✅ Eval Accuracy: {acc:.4f}")

    if show_matrix and class_names:
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()
def predict_live_frame(model, frame, device, class_names, threshold=0.7):
    """
    实时预测一帧图像并返回预测结果和置信度。
    :param model: 已训练的模型
    :param frame: 输入的图像帧
    :param device: 设备，'cuda' 或 'cpu'
    :param class_names: 类别名称列表
    :param threshold: 置信度阈值，低于此值返回 "我看不清..."
    :return: 预测结果和置信度
    """
    model.eval()

    # 预处理图像帧：resize -> 转换为Tensor -> 归一化
    transform = cv2.resize(frame, (224, 224))  # 根据ResNet要求输入大小
    transform = transform.astype(np.float32) / 255.0  # 归一化到[0,1]
    transform = np.transpose(transform, (2, 0, 1))  # HWC -> CHW
    transform = torch.tensor(transform).unsqueeze(0).to(device)  # 增加batch维度并转到device

    # 模型推理
    with torch.no_grad():
        output = model(transform)
        probs = torch.softmax(output, dim=1)  # softmax 归一化为概率
        max_prob, predicted_class = torch.max(probs, 1)

    # 提取预测结果和置信度
    predicted_class = predicted_class.item()
    max_prob = max_prob.item()

    if max_prob < threshold:
        return "我看不清...", max_prob
    else:
        return class_names[predicted_class], max_prob