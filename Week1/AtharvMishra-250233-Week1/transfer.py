import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.optim import lr_scheduler 
import torch.backends.cudnn as cudnn 
import numpy as np 
import torchvision 
from torchvision import datasets, models, transforms 
import matplotlib.pyplot as plt 
import time 
import os 
from PIL import Image 
from tempfile import TemporaryDirectory  

def imshow(inp, title=None):     
    inp = inp.numpy().transpose((1, 2, 0))     
    mean = np.array([0.485, 0.456, 0.406])     
    std = np.array([0.229, 0.224, 0.225])     
    inp = std * inp + mean     
    inp = np.clip(inp, 0, 1)     
    plt.imshow(inp)     
    if title is not None:         
        plt.title(title)     
    plt.pause(0.001)  

cudnn.benchmark = True 
plt.ion()  

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),         
        transforms.ToTensor(),         
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'val': transforms.Compose([
        transforms.Resize(224),         
        transforms.ToTensor(),         
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}  

image_datasets = {'train': datasets.CIFAR10(root='./data', train=True, download=False, transform=data_transforms['train']),
    'val': datasets.CIFAR10(root='./data', train=False, download=False, transform=data_transforms['val'])}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4) for x in ['train', 'val']} 
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']} 
class_names = image_datasets['train'].classes  

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):     
    since = time.time()      
    with TemporaryDirectory() as tempdir:         
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')          
        torch.save(model.state_dict(), best_model_params_path)         
        best_acc = 0.0
        for epoch in range(num_epochs):             
            print(f'Epoch {epoch}/{num_epochs - 1}')             
            print('-' * 10)
            for phase in ['train', 'val']:                 
                if phase == 'train':                     
                    model.train()                  
                else:                     
                    model.eval()                    

                running_loss = 0.0                 
                running_corrects = 0                  

                for inputs, labels in dataloaders[phase]:                     
                    inputs = inputs.to(device)                     
                    labels = labels.to(device)                      
                    optimizer.zero_grad()                      
                    with torch.set_grad_enabled(phase == 'train'):                         
                        outputs = model(inputs)                         
                        _, preds = torch.max(outputs, 1)                         
                        loss = criterion(outputs, labels)                          
                        if phase == 'train':                             
                            loss.backward()                             
                            optimizer.step()                      
                    running_loss += loss.item() * inputs.size(0)                     
                    running_corrects += torch.sum(preds == labels.data)                 
                if phase == 'train':
                    scheduler.step()
                epoch_loss = running_loss / dataset_sizes[phase]                 
                epoch_acc = running_corrects.double() / dataset_sizes[phase]                  
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')                  
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc   
                    torch.save(model.state_dict(), best_model_params_path)              
            print()
        time_elapsed = time.time() - since         
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')         
        print(f'Best val Acc: {best_acc:4f}')          
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))     
    return model  

model_ft = models.resnet18(weights='IMAGENET1K_V1') 
num_ftrs = model_ft.fc.in_features 
model_ft.fc = nn.Linear(num_ftrs, len(class_names))  
model_ft = model_ft.to(device)  

criterion = nn.CrossEntropyLoss()  
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)  
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)  

model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1') 
for param in model_conv.parameters():     
    param.requires_grad = False  

num_ftrs = model_conv.fc.in_features 

model_conv.fc = nn.Linear(num_ftrs, len(class_names))  
model_conv = model_conv.to(device)  

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)  
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)  

model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)  

plt.ioff() 
plt.show()
