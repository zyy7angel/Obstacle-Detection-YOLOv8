import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import io
import tempfile
from tempfile import NamedTemporaryFile
from io import BytesIO
from cv2 import VideoWriter_fourcc
from test import detect_image,detect_video,detect_camera


# 设置证书和密钥文件的路径
os.environ["STREAMLIT_SERVER_CERT"] = "C:/Users/24243/cert.pem"
os.environ["STREAMLIT_SERVER_KEY"] = "C:/Users/24243/key.pem"



# ################设置标题 ###################
st.title('基于YOLOv8的障碍物检测')

# #################主页面###################

# 点击按钮开始检验
# 显示图片/视频/镜头
# st.button('开始检验')


# #################导航栏####################


st.sidebar.title("识别项目设置")

choose = st.sidebar.selectbox('选择检测类型',('图片检测','视频检测','实时摄像头检测'))

# color_selection = ['红色','绿色','蓝色']
# my_color = st.sidebar.radio('障碍物框选颜色',color_selection)


# 选择检测方式
# 1、识别图片

if choose == '图片检测':
    picture_file = st.file_uploader('请上传图片', type=["jpg", "jpeg", "png"])
    if picture_file is not None:
        # 创建临时文件
        tfile = tempfile.NamedTemporaryFile(delete=False)
        bytes_written = tfile.write(picture_file.read())
        output = detect_image(tfile.name)
        tfile.close()
        # # 展示图片
        st.image(output, caption='识别目标', use_column_width=True)

#2、识别视频
# 检查用户是否选择了视频检测选项
if choose == '视频检测':
    video_file = st.file_uploader("请上传视频", type=["mp4", "avi", "mov"])
    if video_file is not None:
        # 显示上传文件的基本信息
        # st.write(f"文件名: {video_file.name}")
        # st.write(f"文件大小: {video_file.size} 字节")

        # 创建临时文件
        tfile = tempfile.NamedTemporaryFile(delete=False)
        # 将上传的视频数据写入临时文件
        bytes_written = tfile.write(video_file.read())
        # st.write(f"写入临时文件的字节数: {bytes_written}")
        # 确认临时文件大小
        tfile.flush()
        # st.write(f"临时文件大小: {os.path.getsize(tfile.name)} 字节")
        # print(tfile.name)

        # # 通过OpenCV打开视频文件
        # cap = cv2.VideoCapture(tfile.name)

        output = detect_video(tfile.name)
        # 使用 st.video 显示视频
        st.video(output)

        tfile.close()



        # # 获取视频的属性
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # # 创建VideoWriter对象，用于保存处理后的视频
        # fourcc = VideoWriter_fourcc(*'avc1')  # 使用'avc1'编解码器
        # out = cv2.VideoWriter('processed_video.mp4', fourcc, fps, (width, height))
        #
        # # 检查视频是否成功打开
        # if not cap.isOpened():
        #     st.write("Error opening video stream or file")
        # else:
        #     # 逐帧读取视频并展示
        #     while cap.isOpened():
        #         success, frame = cap.read()
        #         if success:
        #             # 将BGR格式转换为RGB格式
        #             to_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #             # 转换为灰度图
        #             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #             # 将处理后的帧写入输出视频
        #             out.write(gray_frame)
        #         else:
        #             break
        #     # 释放视频捕获对象和VideoWriter对象
        #     cap.release()
        #     out.release()
        # # 关闭临时文件
        # tfile.close()
        #
        # # 播放处理后的视频
        # st.video('processed_video.mp4')
        #
        # # 提供下载链接
        # st.download_button(
        #     label="Download processed video",
        #     data=open('processed_video.mp4', 'rb').read(),
        #     file_name='processed_video.mp4',
        #     mime='video/mp4'
        # )
        #
        # # 删除临时文件和处理后的视频文件
        # os.remove(tfile.name)
        # os.remove('processed_video.mp4')






# 3、识别实时摄像头
# 创建一个复选框来控制摄像头的启用状态

if choose == '实时摄像头检测':
    enable_camera = st.checkbox("实时摄像头检测")

    if enable_camera:
        detect_camera(enable_camera)

        # # 创建视频捕获对象，0 表示默认摄像头
        # cap = cv2.VideoCapture(0)
        #
        # # 检查摄像头是否成功打开
        # if not cap.isOpened():
        #     st.error("无法打开摄像头")
        # else:
        #     # 创建一个布尔值，用于控制循环
        #     keep_running = True
        #     # 创建一个空的组件，用于更新帧
        #     frame_container = st.empty()
        #
        #     # 循环读取视频帧
        #     while keep_running:
        #         ret, frame = cap.read()
        #         # 如果正确读取帧，ret为True
        #         if not ret:
        #             st.error("无法读取帧")
        #             break
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
