import numpy as np
import torch
import cv2
import os
from torchvision.transforms import v2 as tf
from model import UNet
from data_process import transforms  # 直接使用你訓練時的 transforms
import time

# 基本參數
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')
OUT_CHANNELS = 5
encoder_in_channel_list = [64, 128, 256]
decoder_in_channel_list = [1024, 512, 256, 128]

# 讀取訓練好的模型
model = UNet(3, encoder_in_channel_list, decoder_in_channel_list, OUT_CHANNELS, kernel_size=3, padding='same', bias=False).to(DEVICE)
model.load_state_dict(torch.load('/Users/zhouchenghan/python/Full-shot/Unet/early_stopped_model.pth', map_location = DEVICE))
model.eval()

# 完全使用當時訓練時的 transforms['test']
transform = transforms['test']

COLORMAP = {
    0: (0, 0, 0),         # 黑 - _background_
    1: (0, 255, 0),       # 綠 - Land
    2: (255, 0, 0),       # 紅 - Other
    3: (123, 112, 255),   # 淺紫 - Sky
    4: (135, 206, 250),   # 淺藍 - Sea
}

def label_to_rgb(mask_2d):
    h, w = mask_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in COLORMAP.items():
        rgb[mask_2d == class_id] = color
    return rgb

# 讀取影片路徑
video_path = '/Users/zhouchenghan/python/Full-shot/Unet/video/test.avi'
cap = cv2.VideoCapture(video_path)

# 影片基本資訊
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
alpha = 0.4

# 輸出影片準備
output_path = './predict_video/output_realtime.avi'
os.makedirs('./predict_video', exist_ok=True)
out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 預處理 (完全依照訓練邏輯)
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(input_frame)
    input_tensor = tf.ToDtype(torch.float32, scale=True)(input_tensor)  # 這一行關鍵補上！
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(input_tensor)
        pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
        pred_mask_rgb = label_to_rgb(pred_mask)

    # 調整 mask 大小對應原圖
    if pred_mask_rgb.shape[:2] != frame.shape[:2]:
        pred_mask_rgb = cv2.resize(pred_mask_rgb, (frame.shape[1], frame.shape[0]))

    # 合成遮罩
    blended = cv2.addWeighted(frame, 1 - alpha, cv2.cvtColor(pred_mask_rgb, cv2.COLOR_RGB2BGR), alpha, 0)

    # 輸出即時顯示與存檔
    out_video.write(blended)
    cv2.imshow('Realtime Segmentation', blended)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()