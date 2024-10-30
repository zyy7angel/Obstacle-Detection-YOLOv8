import os
import shutil
import random

# 数据集路径
train_images_dir = 'coco/train/images'
train_labels_dir = 'coco/train/labels'
# val_images_dir = 'coco/val/images'
# val_labels_dir = 'coco/val/labels'

# 创建新的目录
os.makedirs('coco/test_data/images', exist_ok=True)
os.makedirs('coco/test_data/labels', exist_ok=True)
# os.makedirs('coco/val_mini/images', exist_ok=True)
# os.makedirs('coco/val_mini/labels', exist_ok=True)

# 从训练集中随机选择 1000 张图像和标签
train_images = os.listdir(train_images_dir)
random_train_images = random.sample(train_images, 1000)

for image in random_train_images:
    # 复制图像
    shutil.copy(os.path.join(train_images_dir, image), 'coco/test_data/images')
    # 复制对应标签（假设标签与图像同名但扩展名为 .json 或 .txt）
    label_file = os.path.splitext(image)[0] + '.txt'  # 根据实际情况调整扩展名
    shutil.copy(os.path.join(train_labels_dir, label_file), 'coco/test_data/labels')

# # 从验证集中随机选择 500 张图像和标签
# val_images = os.listdir(val_images_dir)
# random_val_images = random.sample(val_images, 500)
#
# for image in random_val_images:
#     # 复制图像
#     shutil.copy(os.path.join(val_images_dir, image), 'coco/val_mini/images')
#     # 复制对应标签
#     label_file = os.path.splitext(image)[0] + '.json'  # 根据实际情况调整扩展名
#     shutil.copy(os.path.join(val_labels_dir, label_file), 'coco/val_mini/labels')
