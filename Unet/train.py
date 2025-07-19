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
EPOCHS = 1000
VERBOSE = 1
OUT_CHANNELS = 5

print (f"Using {DEVICE}")

# def plot(img, mask, pred = None, plot_prediction = False):
    
#     if plot_prediction:
#         fig = plt.figure(figsize = (12, 9))
#         fig.add_subplot(1, 2, 1)
#         plt.imshow(img)
#         plt.title('Original Image')
#         plt.axis(False)
#         fig.add_subplot(1, 2, 2)
#         plt.imshow(pred)
#         plt.title('Predicted Mask')
#         plt.axis(False)
#         plt.show()
#     else:
#         fig = plt.figure()
#         fig.add_subplot(1, 2, 1)
#         plt.imshow(img)
#         plt.title('Original Image')
#         plt.axis(False)
#         fig.add_subplot(1, 2, 2)
#         plt.imshow(mask)
#         plt.title('Original Mask')
#         plt.axis(False)
#         plt.show()

train_dataset = GetDataset(
    'C:/Users/User/Desktop/Semantic_Segmentation/5_data/lars_v1.0.0_images_testttt+NKUST_v2',
    ['train'],
    transforms['train']
)
val_dataset = GetDataset(
    'C:/Users/User/Desktop/Semantic_Segmentation/5_data/lars_v1.0.0_images_testttt+NKUST_v2',
    ['val'],
    transforms['val']
)

print(f"Train dataset contains {len(train_dataset)} instances and validation dataset contains {len(val_dataset)} instances")

# indx = torch.randint(0, len(train_dataset), (1,)).item()
# img, mask = train_dataset[indx]
# print(f"The image is of shape {img.shape} and the mask has shape {mask.shape}")
# img, mask = img.numpy(), mask.numpy()
# img = np.transpose(img, [1, 2, 0])
# plot(img, mask)

train_dataloader = DataLoader(
    dataset = train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    drop_last = False
)

val_dataloader = DataLoader(
    dataset = val_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    drop_last = False
)

print (f"The train dataloader has {len(train_dataloader)} batches and the validation dataloader has {len(val_dataloader)} batches, each of batch size {BATCH_SIZE}")

encoder_in_channel_list = [64, 128, 256]
decoder_in_channel_list = [1024, 512, 256, 128]

model = UNet(3, encoder_in_channel_list, decoder_in_channel_list, OUT_CHANNELS, kernel_size=3, padding='same', bias=False).to(DEVICE)

summary(model, (3, 256, 256))

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
save_dir_working = './working'
os.makedirs(save_dir_working, exist_ok=True)

early_stop = EarlyStop(
    PATIENCE,
    TOLERANCE,
    save_dir_working
)
accuracy_fn = DiceScore(
    num_classes = OUT_CHANNELS,
    include_background = True,
    input_format = 'index'
).to(DEVICE)

run_details = run(
    model,
    DEVICE,
    optimizer,
    loss_fn,
    accuracy_fn,
    train_dataloader,
    val_dataloader,
    early_stop,
    EPOCHS,
    VERBOSE
)

save_dir = './pred'
os.makedirs(save_dir, exist_ok=True)

run_details_path = os.path.join(save_dir, 'metrics.csv')
run_details.to_csv(run_details_path, index=False)

fig = plt.figure(figsize=(12, 4))
fig.add_subplot(1, 2, 1)
plt.plot(
    run_details['epoch'],
    run_details['train_loss'],
    label = 'Train Loss'
)
plt.plot(
    run_details['epoch'],
    run_details['validation_loss'],
    label = 'Validation Loss'
)
plt.title('Loss Plot')
plt.legend()
plt.xlabel('Epoch --->')
plt.ylabel('Loss --->')
fig.add_subplot(1, 2, 2)
plt.plot(
    run_details['epoch'],
    run_details['train_accuracy'],
    label = 'Train Accuracy'
)
plt.plot(
    run_details['epoch'],
    run_details['validation_accuracy'],
    label = 'Validation Accuracy'
)
plt.title('Accuracy Plot')
plt.legend()
plt.xlabel('Epoch --->')
plt.ylabel('Accuracy --->')

plot_path = os.path.join(save_dir, 'train_valid_metrics.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
# plt.show()

torch.save(
    model.state_dict(),
    './working/model.pth'
)
print("save model.pth")