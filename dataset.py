import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import random

class KidneyPatchDataset(Dataset):
    def __init__(self, patch_list_file, transform=None, train=True):
        self.patches = []
        self.labels = []
        
        with open(patch_list_file, 'r') as f:
            for line in f:
                patch_path, label = line.strip().split(',')
                self.patches.append(patch_path)
                self.labels.append(int(label))
        
        self.transform = transform
        self.train = train
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        image_path = self.patches[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(input_size=1024, train=True):
    if train:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def create_dataloaders(patch_list_file, batch_size=16, num_workers=8, train_split=0.8):
    dataset = KidneyPatchDataset(patch_list_file)
    
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_dataset.dataset.transform = get_transforms(train=True)
    val_dataset.dataset.transform = get_transforms(train=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders('train_patches.txt')
    
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Data shape {data.shape}, Target shape {target.shape}")
        print(f"Labels in batch: {target.tolist()}")
        if batch_idx == 2:
            break