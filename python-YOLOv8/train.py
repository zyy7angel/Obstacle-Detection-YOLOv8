# 导入必要的库
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch

# 1. 创建 YOLOv8 模型
model = YOLO('yolov8n.yaml')  # 使用 YOLOv8n

# 2. 数据集路径和训练超参数设置
data_path = 'coco/yolo_coco_dataset.yaml'  # 请将其替换为你的 dataset_coco.yaml 路径
epochs = 5  # 训练轮数
img_size = 640  # 输入图像大小

# 3. 检查 GPU 是否可用，并将模型转移到 GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 4. 开始训练模型
results = model.train(
    data=data_path,  # 数据集配置文件路径
    epochs=epochs,  # 训练的总轮数
    imgsz=img_size,  # 输入图像大小
    batch=32,  # 每批次的图片数量
    name='yolov8_coco_experiment',  # 保存的实验名称
    project='runs/detect',  # 训练结果保存路径
    verbose=True,  # 显示详细训练日志
    device=device  # 使用指定的设备
)

# 5. 可视化训练过程中的损失曲线
# YOLOv8 的训练结果会自动存储损失数据在 results.metrics 中
# 提取训练和验证损失
train_losses = results.metrics['train/loss']
val_losses = results.metrics['val/loss']

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.show()

# 6. 保存训练好的模型
# 训练后的最佳模型会自动保存在 runs/detect/yolov8_coco_experiment/weights/best.pt
model_path = 'runs/detect/yolov8_coco_experiment/weights/best.pt'
print(f'Model saved at: {model_path}')

# 7. 评估模型性能
metrics = model.val(data=data_path, imgsz=img_size, batch=16, device=device)
print(metrics)
