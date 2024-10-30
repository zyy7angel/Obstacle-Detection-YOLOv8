import os

# 创建 YOLOv8 数据集配置文件
yaml_content = f"""
# test_dataset.yaml
train: D:/pythonDoc/python-YOLOv5/coco/train/images      # 训练集图片路径
val: D:/pythonDoc/python-YOLOv5/coco/val/images      # 验证集图片路径
test: D:/pythonDoc/python-YOLOv5/coco/test_data/images      # 测试集图片路径

# 类别名称列表（COCO 数据集有 80 个类别）
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: TV
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
"""

# 保存配置文件
with open(os.path.join('coco', 'yolo_coco_dataset.yaml'), 'w') as f:
    f.write(yaml_content)
