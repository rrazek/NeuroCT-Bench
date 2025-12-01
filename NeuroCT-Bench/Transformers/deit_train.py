# NeuroCT-Bench — transformer training script
# PyTorch + timm implementation

import os, time, random
import numpy as np
import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import pandas as pd
from collections import Counter
from torchvision.transforms import InterpolationMode

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_DIR = os.environ.get('DATA_DIR','dataset')
BATCH_SIZE = int(os.environ.get('BATCH_SIZE',32))
IMG_SIZE = int(os.environ.get('IMG_SIZE',224))
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS',100))
LR = float(os.environ.get('LR',1e-4))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = int(os.environ.get('NUM_WORKERS',4))
SAVE_PATH = 'deit_best.pth'

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.485,0.485], std=[0.229,0.229,0.229])
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.485,0.485], std=[0.229,0.229,0.229])
])

full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
num_samples = len(full_dataset)
train_size = int(0.8 * num_samples)
val_size = num_samples - train_size

g = torch.Generator()
g.manual_seed(SEED)
indices = torch.randperm(num_samples, generator=g).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_ds = Subset(full_dataset, train_indices)
val_full = datasets.ImageFolder(root=DATA_DIR, transform=val_transform)
val_ds = Subset(val_full, val_indices)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

class_names = full_dataset.classes
num_classes = len(class_names)

model = timm.create_model('deit_base_distilled_patch16_224', pretrained=True, num_classes=num_classes)
model.to(DEVICE)

train_labels = [full_dataset[idx][1] for idx in train_indices]
counts = Counter(train_labels)
total = sum(counts.values())
class_weight = {i: total/(len(counts)*v) for i,v in counts.items()}
weights_for_loss = torch.tensor([class_weight[i] for i in range(len(counts))], dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights_for_loss)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

best_val_loss = float('inf')
patience_counter = 0
EARLY_STOPPING_PATIENCE = 10

start_time = time.time()
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    train_loss = running_loss / (len(train_loader.dataset) if len(train_loader.dataset)>0 else 1)

    model.eval()
    all_labels = []
    all_probs = []
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy() if num_classes==2 else torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            val_loss += loss.item() * images.size(0)
    val_loss = val_loss / (len(val_loader.dataset) if len(val_loader.dataset)>0 else 1)
    scheduler.step(val_loss)

    y_val = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val))>1 else float('nan')
    cm = confusion_matrix(y_val, y_pred, labels=[0,1]) if len(np.unique(y_val))>1 else np.array([])
    specificity = (cm.ravel()[0] / (cm.ravel()[0] + cm.ravel()[1])) if cm.size==4 else float('nan')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'epoch': epoch}, 'deit_best.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        break

end_time = time.time()

checkpoint = torch.load('deit_best.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()
all_labels = []
all_probs = []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy() if num_classes==2 else torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

pd.DataFrame({'y_true': all_labels, 'y_prob': all_probs, 'y_pred': (np.array(all_probs)>0.5).astype(int)}).to_csv('deit_val_predictions.csv', index=False)
