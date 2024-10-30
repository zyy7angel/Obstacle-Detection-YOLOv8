import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO('runs/detect/yolov8_coco_experiment/weights/best.pt')

# 在验证集上进行评估
results = model.val(data='coco/yolo_coco_dataset.yaml')

# 打印评估指标
metrics = results['metrics']
print(f'Precision: {metrics["precision"]:.4f}')
print(f'Recall: {metrics["recall"]:.4f}')
print(f'F1 Score: {metrics["f1"]:.4f}')
print(f'mAP@0.5: {metrics["mAP_0.5"]:.4f}')
print(f'mAP@0.5:0.95: {metrics["mAP_0.5:0.95"]:.4f}')

# 可视化指标
metric_names = ['Precision', 'Recall', 'F1 Score', 'mAP@0.5', 'mAP@0.5:0.95']
metric_values = [
    metrics['precision'],
    metrics['recall'],
    metrics['f1'],
    metrics['mAP_0.5'],
    metrics['mAP_0.5:0.95']
]

plt.figure(figsize=(10, 5))
sns.barplot(x=metric_names, y=metric_values)
plt.title('Model Evaluation Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()
