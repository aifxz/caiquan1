# train.py
import torch
from 推理代码 import FineTuneResNet, setup_training_stage, train_model
from 数据处理 import get_device, load_data, preview_images

if __name__ == "__main__":
    data_dir = r"E:\machinelearning\datacollection\caiquan"  # 数据集路径

    # 获取设备
    device = get_device()

    # 加载数据
    hands = None  # 训练时一般不用手部裁剪，直接设为None
    train_loader, val_loader, class_names = load_data(data_dir, hands)

    # 预览一下图像
    preview_images(train_loader, class_names)

    # 初始化模型
    model = FineTuneResNet(num_classes=len(class_names)).to(device)

    # 第一阶段：训练全连接层
    optimizer = setup_training_stage(model, stage="fc", base_lr=1e-3)
    train_model(model, train_loader, val_loader, optimizer, device, epochs=10, save_path="model/fc")

    # 第二阶段：训练 layer4
    optimizer = setup_training_stage(model, stage="layer4", base_lr=1e-5)
    train_model(model, train_loader, val_loader, optimizer, device, epochs=5, save_path="model/layer4")

    print("✅ 训练完成！模型已保存。")
