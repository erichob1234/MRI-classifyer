import os
import torch 
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import torch.optim as optim
from torch.cuda import amp

print("torch.cuda.is_available():", torch.cuda.is_available())
print("CUDA build:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))

#initialize the random seeds
def seed_everything(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe even if no GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#seed everything
seed_everything(42)

#define path to data
datapath = Path("data")

#normalize mri scans using per image z score to eliminate brightness variation
class perimagez:
    def __call__(self, tensor:torch.Tensor) -> torch.Tensor:
        mu = tensor.mean(dim=(1, 2), keepdim=True) 
        sigma = tensor.std(dim=(1, 2), unbiased=False, keepdim=True) 
        return (tensor - mu) / (sigma + 1e-6)
    
#making callable transformation object
transformations = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((224, 224)),transforms.ToTensor(),perimagez()])

#make datasets, imagefolder auto labels the images with a class index, makes it an object
train_ds = datasets.ImageFolder(datapath / "train", transform=transformations)
val_ds   = datasets.ImageFolder(datapath / "val",   transform=transformations)
test_ds  = datasets.ImageFolder(datapath / "test",  transform=transformations)

#dataloaders to make mini batches and shuffle data
batchsize = 32
NUM_WORKERS = 0 

pin = torch.cuda.is_available()

train_dl = DataLoader(train_ds, batch_size=batchsize, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin, drop_last=False)
val_dl   = DataLoader(val_ds, batch_size=batchsize, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
test_dl  = DataLoader(test_ds, batch_size=batchsize, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

#making the model
def resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    old = model.conv1
    model.conv1 = nn.Conv2d(in_channels=1, out_channels=old.out_channels, kernel_size=old.kernel_size, stride=old.stride,padding=old.padding,bias=False)
    #initalizing activation relu
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    #replacing resnet final fully connected layer default output with the four mri scan classes
    model.fc=nn.Linear(model.fc.in_features, num_classes)
    #initialize final layer weight and zero bias
    nn.init.normal_(model.fc.weight, 0, 0.01)
    nn.init.zeros_(model.fc.bias)
    return model


#running stuff on gpu through cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # autotunes convs for fixed 224x224

num_classes = len(train_ds.classes)
model = resnet18(num_classes).to(device)

#calculated class weights
weights = [0.9482, 1.3145, 0.8850, 0.9482]
#converting to tensor
class_weights = torch.tensor(weights, dtype=torch.float32)
#choosing loss function
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
#optimizer
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

#setting up mixed precision training for gpu
use_amp = (device.type == "cuda")
scaler = torch.amp.GradScaler(enabled=use_amp)

#training loop
fullpass = 10

for epoch in range(1, fullpass + 1):
    model.train()
    running_loss, running_corrects, total = 0.0, 0, 0

    for xb, yb in train_dl: #everytime you iterate dl it gives you a batch with the image and label because of dataloader
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        #clear gradients
        optimizer.zero_grad(set_to_none=True)

        # forward pass
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(xb)
            #computing loss
            loss = criterion(logits, yb)

        # backward + optimization step
        if use_amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            #computing gradient of loss with respect to all the weights with backpropogation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            #update weights 
            optimizer.step()

        # track stats
        batch_size = xb.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += (logits.argmax(1) == yb).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total

    print(f"Epoch {epoch:02d}/{fullpass} | "
          f"loss {epoch_loss:.4f} | acc {epoch_acc:.3f}")