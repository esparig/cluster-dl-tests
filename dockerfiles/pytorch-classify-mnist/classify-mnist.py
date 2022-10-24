from datetime import datetime
from functools import wraps
import time

import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

def print_timing(func):
    '''
    create a timing decorator function
    use
    @print_timing
    just above the function you want to time
    '''
    @wraps(func)  # improves debugging
    def wrapper(*arg):
        date_time = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'[{date_time}]')
        start = time.perf_counter()  # needs python3.3 or higher
        result = func(*arg)
        end = time.perf_counter()
        fs = '{} took {:.3f} seconds'
        print(fs.format(func.__name__, (end - start)))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return result
    return wrapper

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class MnistModel(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        # output layer
        self.linear2 = nn.Linear(hidden_size, out_size)
        
    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear2(out)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
 
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

@print_timing       
def run():
    dataset = MNIST(root='data', train=True, download=False, transform=ToTensor())
    val_size = 10000
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    len(train_ds), len(val_ds)
    
    batch_size=128
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
    
    input_size = 784
    hidden_size = 32 # you can change this
    num_classes = 10
    
    model = MnistModel(input_size, hidden_size=32, out_size=num_classes)

    for t in model.parameters():
        print(t.shape)
    
    for images, labels in train_loader:
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        print('Loss:', loss.item())
        break
    print('outputs.shape : ', outputs.shape)
    print('Sample outputs :\n', outputs[:2].data)
    
    device = get_default_device()
    print(device)
    
    for images, labels in train_loader:
        print(images.shape)
        images = to_device(images, device)
        print(images.device)
        break
    
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    
    for xb, yb in val_loader:
        print('xb.device:', xb.device)
        print('yb:', yb)
        break
    
    # Model (on GPU)
    model = MnistModel(input_size, hidden_size=hidden_size, out_size=num_classes)
    to_device(model, device)
    
    history = [evaluate(model, val_loader)]
    print(history)
    
    history += fit(5, 0.5, model, train_loader, val_loader)
    print(history)
    
    history += fit(5, 0.1, model, train_loader, val_loader)
    print(history)

if __name__ == "__main__":
    run()
