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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_RATIO = 0.8
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
PATIENCE = 15
TOLERANCE = 1e-3
EPOCHS = 150
VERBOSE = 1
OUT_CHANNELS = 5

print (f"Using {DEVICE}")

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

def plot(img, pred = None, plot_prediction = False):
    
    if plot_prediction:
        fig = plt.figure(figsize = (12, 9))
        fig.add_subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis(False)
        fig.add_subplot(1, 2, 2)
        plt.imshow(pred)
        plt.title('Predicted Mask')
        plt.axis(False)
        plt.show()

test_dataset = test_GetDataset(
    'C:/Users/User/Desktop/images',
    ['images'],
    transforms['test']
)

test_dataloader = DataLoader(
    dataset = test_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False,
    drop_last = False
)

encoder_in_channel_list = [64, 128, 256]
decoder_in_channel_list = [1024, 512, 256, 128]

model = UNet(3, encoder_in_channel_list, decoder_in_channel_list, OUT_CHANNELS, kernel_size=3, padding='same', bias=False).to(DEVICE)
summary(model, (3, 256, 256))
model.load_state_dict(torch.load('/Users/zhouchenghan/python/Full-shot/Unet/early_stopped_model.pth', map_location = DEVICE))

save_dir_img = './pred/image'
os.makedirs(save_dir_img, exist_ok=True)

save_dir_mask = './pred/mask'
os.makedirs(save_dir_mask, exist_ok=True)

save_dir_mask_rgb = './pred/mask_rgb'
os.makedirs(save_dir_mask_rgb, exist_ok=True)

model.eval()
all_imgs = []
all_preds = []

with torch.no_grad():
    for img_batch in tqdm(test_dataloader, desc="testing"):
        img_batch = img_batch.to(DEVICE)
        pred_batch = model(img_batch)
        pred_batch = pred_batch.argmax(dim=1)

        # [img_0, img_1, ..., img_(B-1)]
        all_imgs.extend(img_batch.cpu())
        all_preds.extend(pred_batch.cpu())


for i, (img_tensor, pred_tensor) in tqdm(enumerate(zip(all_imgs, all_preds)), total = len(all_imgs), desc = "Saving results"):
    pred = pred_tensor.numpy().astype(np.uint8)
    img = img_tensor.numpy()
    img = np.transpose(img, [1, 2, 0])
    img = (img * 255).astype(np.uint8)

    img_resized = Image.fromarray(img).resize((640, 512), resample=Image.BILINEAR)
    img_resized.save(os.path.join(save_dir_img, f"img_{i}.png"))

    pred_resized = Image.fromarray(pred, mode='L').resize((640, 512), resample=Image.NEAREST)
    pred_resized.save(os.path.join(save_dir_mask, f"img_{i}.png"))

    pred_rgb = label_to_rgb(np.array(pred_resized))
    pred_rgb_img = Image.fromarray(pred_rgb)
    pred_rgb_img.save(os.path.join(save_dir_mask_rgb, f"img_{i}.png"))

    # plot(np.array(img_resized), pred_rgb, True)
