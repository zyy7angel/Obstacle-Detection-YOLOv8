import os
import json

# 设定路径
annotation_file_path = 'annotations/instances_train2017.json'  # JSON标注文件路径
label_save_dir = 'coco/train/labels'  # 标签文件的存储目录

# 如果标签目录不存在，则创建
os.makedirs(label_save_dir, exist_ok=True)

# 加载标注文件
with open(annotation_file_path, 'r') as f:
    coco_data = json.load(f)

# 创建一个字典，按照 image_id 组织标注信息
image_annotations = {}

# 原始类别 ID 到新的类别 ID 映射
category_mapping = {
    1: 0,  # person
    2: 1,  # bicycle
    3: 2,  # car
    4: 3,  # motorcycle
    5: 4,  # airplane
    6: 5,  # bus
    7: 6,  # train
    8: 7,  # truck
    9: 8,  # boat
    10: 9,  # traffic light
    # 省略其他类别...
    90: 79  # toothbrush
}

for ann in coco_data['annotations']:
    image_id = ann['image_id']
    category_id = ann['category_id']
    bbox = ann['bbox']

    # 使用映射字典将类别 ID 转换为新的 ID
    new_category_id = category_mapping.get(category_id, -1)  # 默认使用 -1 表示未知类别
    if new_category_id == -1:
        continue  # 如果类别未知，跳过该标注

    # 将 annotation 信息组织为 [new_category_id, bbox_x, bbox_y, bbox_width, bbox_height]
    annotation_info = [new_category_id] + bbox

    # 添加 annotation 到对应的 image_id 下
    if image_id not in image_annotations:
        image_annotations[image_id] = []
    image_annotations[image_id].append(annotation_info)

# 将每张图片的标注信息保存为单独的标签文件
for image in coco_data['images']:
    image_id = image['id']
    file_name = image['file_name']
    img_width = image['width']
    img_height = image['height']

    # 定义标签文件的存储路径，使用图片 ID 或名称（不带扩展名）
    label_path = os.path.join(label_save_dir, f"{os.path.splitext(file_name)[0]}.txt")

    # 获取该图片的所有标注信息
    annotations = image_annotations.get(image_id, [])

    # 将每个标注信息写入标签文件
    with open(label_path, 'w') as f:
        for ann in annotations:
            new_category_id, bbox_x, bbox_y, bbox_width, bbox_height = ann
            # 标准化 bbox 值
            norm_x = bbox_x / img_width
            norm_y = bbox_y / img_height
            norm_width = bbox_width / img_width
            norm_height = bbox_height / img_height

            # 计算 YOLO 格式的中心点坐标 (x_center, y_center)
            x_center = norm_x + norm_width / 2
            y_center = norm_y + norm_height / 2

            # 写入标准化的 YOLO 格式 (class_id, x_center, y_center, width, height)
            f.write(f"{new_category_id} {x_center} {y_center} {norm_width} {norm_height}\n")

    print(f"Saved label file for {file_name}")
