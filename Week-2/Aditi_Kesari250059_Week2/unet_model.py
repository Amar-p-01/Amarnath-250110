import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}")


## 1. Custom Dataset Class
#The dataset contains a main image and a `masks` folder with individual binary masks for every single nucleus in that image. The `DataScienceBowlDataset` class reads the main image,
# iterates through the `masks` folder, and merges all individual masks into one unified target mask using `np.maximum`.
class DataScienceBowlDataset(Dataset):
    def __init__(self, root_dir, img_size=128):
        self.root_dir = root_dir
        self.image_ids = next(os.walk(root_dir))[1]
        self.img_size = img_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_folder = os.path.join(self.root_dir, img_id)
        
        # 1. Read and resize the main image
        img_path = os.path.join(img_folder, 'images', img_id + '.png')
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # 2. Combine all individual masks into one
        mask_dir = os.path.join(img_folder, 'masks')
        combined_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        for mask_name in next(os.walk(mask_dir))[2]:
            mask_path = os.path.join(mask_dir, mask_name)
            single_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            single_mask = cv2.resize(single_mask, (self.img_size, self.img_size))
            combined_mask = np.maximum(combined_mask, single_mask)
            
        # Normalize pixel values to [0, 1]
        combined_mask = combined_mask / 255.0
        image = image / 255.0
        
        # Format for PyTorch: (Channels, Height, Width)
        image = np.transpose(image, (2, 0, 1))
        
        image_tensor = torch.tensor(image, dtype=torch.float32)
        mask_tensor = torch.tensor(combined_mask, dtype=torch.float32).unsqueeze(0)
        
        return image_tensor, mask_tensor


## 2. U-Net Model Definition
#The architecture consists of an encoder (contracting path) to capture context and a decoder (symmetric expanding path) that enables precise localization. 
#Skip connections concatenate feature maps from the encoder directly to the decoder to recover spatial details lost during pooling.

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Down
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        # Bottleneck
        bn = self.bottleneck(self.pool4(d4))

        # Up with Skip Connections
        u1 = self.up1(bn)
        u1 = self.conv_up1(torch.cat([u1, d4], dim=1))
        
        u2 = self.up2(u1)
        u2 = self.conv_up2(torch.cat([u2, d3], dim=1))
        
        u3 = self.up3(u2)
        u3 = self.conv_up3(torch.cat([u3, d2], dim=1))
        
        u4 = self.up4(u3)
        u4 = self.conv_up4(torch.cat([u4, d1], dim=1))

        return self.outc(u4)


## 3. Initialization and Dataloaders
#We instantiate the dataset, perform an 80/20 train/validation split, and initialize the U-Net model. We use `BCEWithLogitsLoss`, which combines a Sigmoid layer and the Binary Cross
# Entropy Loss in one single, numerically stable class.

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 30
IMG_SIZE = 128
DATA_DIR = 'data-science-bowl-2018/stage1_train'

# Prepare Data
dataset = DataScienceBowlDataset(root_dir=DATA_DIR, img_size=IMG_SIZE)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize Model, Loss, and Optimizer
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4 Training Loop
train_losses = []

print("Starting Training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for images, masks in train_loader:
        # Move tensors to GPU
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}")

# Plotting the training loss
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()


# 5 Model Evaluation: 
model.eval()

# Grab a single batch from the validation loader
val_images, val_masks = next(iter(val_loader))

# Select the first image in the batch
sample_image = val_images[0].unsqueeze(0).to(device)
ground_truth = val_masks[0].squeeze().numpy()

with torch.no_grad():
    # Forward pass
    output = model(sample_image)
    # Apply sigmoid to convert logits to probabilities, then threshold
    pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    pred_mask = (pred_mask > 0.5).astype(np.float32)

# Convert the image back to HWC format for matplotlib plotting
display_image = sample_image.squeeze().cpu().numpy().transpose(1, 2, 0)

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.imshow(display_image)
ax1.set_title('Original Image')
ax1.axis('off')

ax2.imshow(ground_truth, cmap='gray')
ax2.set_title('Ground Truth Mask')
ax2.axis('off')

ax3.imshow(pred_mask, cmap='gray')
ax3.set_title('Predicted Mask')
ax3.axis('off')

plt.tight_layout()
plt.show()
torch.save(model.state_dict(), 'unet_model.pth')
