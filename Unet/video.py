# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import torch
# from torch import optim, nn
# from torchvision.transforms import v2 as tf
# from torch.utils.data import Dataset, DataLoader
# from torchsummary import summary
# from torchmetrics.segmentation import DiceScore
# import os
# from PIL import Image
# from tqdm.auto import tqdm
# from data_process import GetDataset, test_GetDataset, transforms
# from model import UNet, EarlyStop, run
# import cv2

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# TRAIN_RATIO = 0.8
# BATCH_SIZE = 16
# LEARNING_RATE = 1e-4
# PATIENCE = 15
# TOLERANCE = 1e-3
# EPOCHS = 150
# VERBOSE = 1

# COLORMAP = {
#     0: (255, 0, 0),       # 紅 - Ship
#     1: (0, 255, 0),       # 綠 - Land
#     2: (0, 0, 255),       # 藍 - Dock
#     3: (255, 255, 0),     # 黃 - Buoy
#     4: (255, 0, 255),     # 紫 - Bridge
#     5: (0, 0, 0),         # 黑 - Other
#     6: (0, 255, 255),     # 青 - Sky
#     7: (128, 128, 128),   # 灰 - Sea
#     8: (255, 165, 0),     # 橘 - Reef
# }

# def label_to_rgb(mask_2d):
#     h, w = mask_2d.shape
#     rgb = np.zeros((h, w, 3), dtype=np.uint8)
#     for class_id, color in COLORMAP.items():
#         rgb[mask_2d == class_id] = color
#     return rgb

# class VideoDataset(Dataset):
#     def __init__(self, video_path, transforms=None, max_frames=None):
#         self.video_path = video_path
#         self.transforms = transforms
#         self.frames = []
#         self._convert_dtype = tf.ToDtype(torch.float32, scale=True)
#         self._normalize = tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
#         cap = cv2.VideoCapture(self.video_path)
#         success, frame = cap.read()
#         count = 0

#         while success:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             self.frames.append(frame)
#             success, frame = cap.read()
#             count += 1
#             if max_frames and count >= max_frames:
#                 break
#         cap.release()

#     def __len__(self):
#         return len(self.frames)

#     def __getitem__(self, index):
#         img = self.frames[index]
#         if self.transforms:
#             img = self.transforms(img)
#         img = self._convert_dtype(img)
#         return img, index  # 同時回傳 index 作為儲存序號

# video_path = 'C:/Users/User/Desktop/drive-download-20250612T082906Z-1-001/20250610_112946.mp4'

# test_dataset = VideoDataset(
#     video_path,
#     transforms['test']
# )

# test_dataloader = DataLoader(
#     dataset=test_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     drop_last=False
# )

# encoder_in_channel_list = [64, 128, 256]
# decoder_in_channel_list = [1024, 512, 256, 128]

# model = UNet(3, encoder_in_channel_list, decoder_in_channel_list, 10, kernel_size=3, padding='same', bias=False).to(DEVICE)
# summary(model, (3, 256, 256))
# model.load_state_dict(torch.load('./working/early_stopped_model.pth', map_location = DEVICE))


# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = 20
# frame_size = (256, 256)

# predict_video_mask_file = './predict_video/png'
# os.makedirs(predict_video_mask_file, exist_ok=True)

# # 改用 AVI 格式
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video_out_path = './predict_video/output_pred_video.avi'
# video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, frame_size)

# alpha = 0.4

# # 動態抓取解析度
# sample_frame = test_dataset.frames[0]
# height, width = sample_frame.shape[:2]
# frame_size = (width, height)

# model.eval()
# with torch.no_grad():
#     for img_batch, indices in tqdm(test_dataloader, desc="Predicting"):
#         img_batch = img_batch.to(DEVICE)
#         pred_batch = model(img_batch)
#         pred_batch = pred_batch.argmax(dim=1).cpu().numpy()

#         for i, pred_mask in enumerate(pred_batch):
#             index = indices[i].item()
#             rgb_mask = label_to_rgb(pred_mask)
#             original_rgb = test_dataset.frames[index]

#             # 自動調整解析度
#             if rgb_mask.shape[:2] != original_rgb.shape[:2]:
#                 rgb_mask = cv2.resize(rgb_mask, (original_rgb.shape[1], original_rgb.shape[0]))

#             blended = cv2.addWeighted(original_rgb, 1 - alpha, rgb_mask, alpha, 0)
#             blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

#             # # 儲存單張 blended 結果
#             # Image.fromarray(blended).save(f'{predict_video_mask_file}/{index:04d}.png')

#             # 確保資料格式正確
#             if blended_bgr.shape[:2][::-1] != frame_size:
#                 blended_bgr = cv2.resize(blended_bgr, frame_size)

#             video_writer.write(blended_bgr)

# video_writer.release()
# print("影片儲存完成：", video_out_path)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import optim, nn
from torchvision.transforms import v2 as tf
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torchmetrics.segmentation import DiceScore
import os
from PIL import Image
from tqdm.auto import tqdm
from data_process import GetDataset, test_GetDataset, transforms
from model import UNet, EarlyStop, run
import cv2
import glob

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_RATIO = 0.8
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
PATIENCE = 15
TOLERANCE = 1e-3
EPOCHS = 150
VERBOSE = 1
OUT_CHANNELS = 5

encoder_in_channel_list = [64, 128, 256]
decoder_in_channel_list = [1024, 512, 256, 128]

model = UNet(3, encoder_in_channel_list, decoder_in_channel_list, OUT_CHANNELS, kernel_size=3, padding='same', bias=False).to(DEVICE)
model.load_state_dict(torch.load('/Users/zhouchenghan/python/Full-shot/Unet/early_stopped_model.pth', map_location = DEVICE))
model.eval()

# COLORMAP = {
#     0: (0, 0, 0),         # 黑 - _background_
#     1: (255, 0, 0),       # 紅 - Ship
#     2: (0, 255, 0),       # 綠 - Land
#     3: (0, 0, 255),       # 藍 - Dock
#     4: (255, 255, 0),     # 黃 - Buoy
#     5: (255, 0, 255),     # 紫 - Bridge
#     6: (128, 128, 128),   # 灰 - Other
#     7: (123, 112, 255),     # 淺紫 - Sky
#     8: (135, 206, 250),   # 淺藍 - Sea
#     9: (255, 165, 0),     # 橘 - Reef
# }

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

class VideoDataset(Dataset):
    def __init__(self, video_path, transforms=None, max_frames=None):
        self.video_path = video_path
        self.transforms = transforms
        self.frames = []
        self._convert_dtype = tf.ToDtype(torch.float32, scale=True)
        self._normalize = tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        cap = cv2.VideoCapture(self.video_path)
        success, frame = cap.read()
        count = 0

        while success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames.append(frame)
            success, frame = cap.read()
            count += 1
            if max_frames and count >= max_frames:
                break
        cap.release()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        img = self.frames[index]
        if self.transforms:
            img = self.transforms(img)
        img = self._convert_dtype(img)
        return img, index  # 同時回傳 index 作為儲存序號

def process_video_file(video_path):
    print(f"/n開始處理影片：{video_path}")
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 建立輸出路徑
    output_folder = f'./predict_video/{video_name}'
    os.makedirs(output_folder, exist_ok=True)

    # 使用 OpenCV 先抓取原始 fps
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps == 0 or np.isnan(fps):
        fps = 20  # 預設 fallback 避免抓不到

    # Dataset 準備
    test_dataset = VideoDataset(video_path, transforms['test'])
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 取樣一張 frame 取得大小
    sample_frame = test_dataset.frames[0]
    height, width = sample_frame.shape[:2]
    frame_size = (width, height)

    # 建立輸出影片 writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_path = os.path.join(output_folder, f'{video_name}_predict.avi')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    alpha = 0.4
    model.eval()
    with torch.no_grad():
        for img_batch, indices in tqdm(test_dataloader, desc=f"Predicting {video_name}"):
            img_batch = img_batch.to(DEVICE)
            pred_batch = model(img_batch)
            pred_batch = pred_batch.argmax(dim=1).cpu().numpy()

            for i, pred_mask in enumerate(pred_batch):
                index = indices[i].item()
                rgb_mask = label_to_rgb(pred_mask)
                original_rgb = test_dataset.frames[index]

                if rgb_mask.shape[:2] != original_rgb.shape[:2]:
                    rgb_mask = cv2.resize(rgb_mask, (original_rgb.shape[1], original_rgb.shape[0]))

                blended = cv2.addWeighted(original_rgb, 1 - alpha, rgb_mask, alpha, 0)
                blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

                if blended_bgr.shape[:2][::-1] != frame_size:
                    blended_bgr = cv2.resize(blended_bgr, frame_size)

                video_writer.write(blended_bgr)

    video_writer.release()
    print(f"儲存完成：{output_video_path}")

video_folder = '/Users/zhouchenghan/python/Full-shot/Unet/video'
video_files = glob.glob(os.path.join(video_folder, '*.avi')) + glob.glob(os.path.join(video_folder, '*.avi'))

for video_path in video_files:
    process_video_file(video_path)