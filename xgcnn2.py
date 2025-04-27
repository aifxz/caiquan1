import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader

# ✅ 定义模型（冻结 + 只训练分类头）
class FineTuneResNet(nn.Module):
    def __init__(self, num_classes):
        super(FineTuneResNet, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in self.resnet.parameters():
            param.requires_grad = False  # 冻结全部参数

        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# ✅ 训练函数
def train_model(train_loader, val_loader, class_names, device, epochs=10):
    model = FineTuneResNet(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.resnet.fc.parameters(), lr=0.001)

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
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Acc: {acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "finetuned_resnet.pth")
    return model
