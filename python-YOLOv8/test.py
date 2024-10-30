import torch
from ultralytics import YOLO
from PIL import Image
from cv2 import VideoWriter_fourcc
import io
import streamlit as st


# 加载模型
model = YOLO('D:\\pythonDoc\\python-YOLOv5\\runs\\detect\\yolov8_coco_experiment4\\weights\\best.pt')


def detect_image(image_path):
    img = Image.open(image_path)
    results = model(img)
    # 创建一个字节流来保存绘制了检测框的图像
    byte_stream = io.BytesIO()

    for result in results:
        # 使用 result.plot() 方法绘制检测结果，它返回一个 numpy.ndarray
        annotated_img = result.plot()
        # 检查 annotated_img 是否为 None
        if annotated_img is not None:
            # 将 numpy.ndarray 转换为 PIL.Image 对象
            img_pil = Image.fromarray(annotated_img)

            # 保存 PIL.Image 对象到 byte_stream
            img_pil.save(byte_stream, format='PNG')
        else:
            print("Plot method returned None, skipping save.")

        # 将 byte_stream 设置为文件开始的位置
        byte_stream.seek(0)

        # 返回 byte_stream 以便在 Streamlit 中显示
        return byte_stream


# # 调用图片识别函数
# detect_image(r"C:\Users\24243\Pictures\Figure_2.png")

import cv2
def detect_camera(enable_camera):
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        st.error("无法打开摄像头")
    else:
        # 创建一个布尔值，用于控制循环
        keep_running = True
        # 创建一个空的组件，用于更新帧
        frame_container = st.empty()

        # 循环读取视频帧
        while keep_running:
            ret, frame = cap.read()
            # 如果正确读取帧，ret为True
            if not ret:
                st.error("无法读取帧")
                break
            # 转换为YOLO输入格式并进行预测
            results = model(frame)

            # 处理并显示结果
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes.data:
                        x1, y1, x2, y2, conf, cls = box
                        # 绘制边界框和标签
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f'{int(cls)} {conf:.2f}', (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 使用frame_container更新帧
            # 将BGR图像转换为RGB图像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_container.image(frame_rgb, caption='Processed Frame', use_column_width=True)

        #创建一个单选框，用于控制是否继续读取摄像头
            if enable_camera is None:
                keep_running = False

                # 释放视频捕获对象
    cap.release()


        #
    #         # 对帧进行处理，例如转换为灰度图
    #         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    #         # 将处理后的帧转换为Streamlit可以显示的格式
    #         # 将numpy数组转换为PIL图像，然后转换为适合st.image的格式
    #         img_bytes = cv2.imencode('.png', gray_frame)[1].tobytes()
    #         image = Image.open(io.BytesIO(img_bytes))
    #
    #         # 使用frame_container更新帧
    #         frame_container.image(image, caption='Processed Frame', use_column_width=True)
    #
    #         # 创建一个单选框，用于控制是否继续读取摄像头
    #         if enable_camera is None:
    #             keep_running = False
    #
    #         # 释放视频捕获对象
    # cap.release()



    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #
    #     # 转换为YOLO输入格式并进行预测
    #     results = model(frame)
    #
    #     # 处理并显示结果
    #     for result in results:
    #         boxes = result.boxes
    #         if boxes is not None:
    #             for box in boxes.data:
    #                 x1, y1, x2, y2, conf, cls = box
    #                 # 绘制边界框和标签
    #                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    #                 cv2.putText(frame, f'{int(cls)} {conf:.2f}', (int(x1), int(y1) - 10),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # # 显示带检测框的实时视频
        # cv2.imshow("Camera Detection", frame)
        #
        # # 按 'q' 键退出
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # if enable_camera is None:
        #     break



    cap.release()
    cv2.destroyAllWindows()


# # 调用实时摄像头识别函数
# detect_camera()


import cv2


def detect_video(video_path, output_path='output_video.mp4', conf_threshold=0.5, fixed_width=800, fixed_height=600):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的宽、高、帧率信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = VideoWriter_fourcc(*'avc1')

    # 初始化视频写入对象
    out = cv2.VideoWriter(output_path, fourcc, fps, (fixed_width, fixed_height))

    # # 设置显示窗口的名称和固定大小
    # cv2.namedWindow("Video Detection", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Video Detection", fixed_width, fixed_height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 进行检测
        results = model(frame)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes.data:
                    x1, y1, x2, y2, conf, cls = box

                    if conf < conf_threshold:
                        continue

                    box_thickness = max(1, int((x2 - x1) / 100))
                    font_scale = float(max(0.5, (x2 - x1) / 200))

                    cv2.rectangle(frame,
                                  (int(x1), int(y1)),
                                  (int(x2), int(y2)),
                                  (0, 255, 0),
                                  thickness=box_thickness)

                    label = f'{int(cls)} {conf:.2f}'
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                    )
                    cv2.rectangle(frame,
                                  (int(x1), int(y1) - text_height - baseline),
                                  (int(x1) + text_width, int(y1)),
                                  (0, 255, 0),
                                  -1)
                    cv2.putText(frame,
                                label,
                                (int(x1), int(y1) - baseline),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale,
                                (0, 0, 0),
                                thickness=1)

        # 调整视频帧大小以适应固定窗口
        frame_resized = cv2.resize(frame, (fixed_width, fixed_height))
        out.write(frame_resized)
        # cv2.imshow("Video Detection", frame_resized)
        #
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    # cv2.destroyAllWindows()

    return output_path




# 调用视频识别函数，设置固定窗口大小，例如 800x600
detect_video('test/test_video.mp4', fixed_width=800, fixed_height=600)


