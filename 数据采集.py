import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime

# 英文路径设置
base_dir = r"E:\machinelearning\datacollection\caiquan"
labels = ['rock', 'paper', 'scissors']
current_label = 'rock'
counter = 0

# 创建目录
for label in labels:
    os.makedirs(os.path.join(base_dir, label), exist_ok=True)

cap = cv2.VideoCapture(0)

print("按 'r' 切换 rock，'p' 切换 paper，'s' 切换 scissors，'c' 保存，'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    text = f"Label: {current_label}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        current_label = 'rock'
    elif key == ord('p'):
        current_label = 'paper'
    elif key == ord('s'):
        current_label = 'scissors'
    elif key == ord('c'):
        label_dir = os.path.join(base_dir, current_label)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(label_dir, f"{current_label}_{timestamp}.jpg")
        try:
            saved = cv2.imwrite(filename, frame)
            if saved:
                print(f"[保存成功] {filename}")
            else:
                print(f"[保存失败] {filename}")
        except Exception as e:
            print(f"[异常] {filename} - {e}")
        counter += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ========= 退出后，自动展示图片缩略图 =========
print("📸 正在加载图像预览...")

fig, axs = plt.subplots(len(labels), 5, figsize=(12, 6))
for i, label in enumerate(labels):
    label_dir = os.path.join(base_dir, label)
    images = os.listdir(label_dir)
    for j in range(5):
        ax = axs[i, j]
        ax.axis('off')
        if j < len(images):
            img_path = os.path.join(label_dir, images[j])
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.set_title(f"{label}_{j}")

plt.tight_layout()
plt.show()

# ========= 自动统计每类图片数量 =========
print("统计每类图片数量：")
for label in labels:
    label_dir = os.path.join(base_dir, label)
    image_count = len(os.listdir(label_dir))
    print(f"{label}: {image_count} 张图片")
