import os

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms


def load_image(train_dir, val_dir, test_dir):
    resize = transforms.Resize(size=(224, 224))
    h_flip = transforms.RandomHorizontalFlip(p=0.25)
    v_flip = transforms.RandomVerticalFlip(p=0.25)
    rotate = transforms.RandomRotation(degrees=15)

    train_transforms = transforms.Compose([resize, h_flip, v_flip, rotate, transforms.ToTensor()])
    test_transforms = transforms.Compose([resize, transforms.ToTensor()])
    print("[INFO] loading the training and validation dataset...")
    train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = ImageFolder(root=val_dir, transform=test_transforms)
    test_dataset = ImageFolder(root=test_dir, transform=test_transforms)
    print("[INFO] training dataset contains {} samples...".format(len(train_dataset)))
    print("[INFO] validation dataset contains {} samples...".format(len(val_dataset)))
    print("[INFO] test dataset contains {} samples...".format(len(test_dataset)))

    print("[INFO] creating training and validation set dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=train_dataset.__len__(), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_dataset.__len__())
    test_loader = DataLoader(test_dataset, batch_size=test_dataset.__len__())

    return train_loader, val_loader, test_loader
