import albumentations as A
import numpy as np
import PIL.Image
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader

import os

from data import AntsBeesDataset


if __name__ == "__main__":

    data_dir = "./hymenoptera_data"

    # Just normalization for validation
    data_transforms = A.Compose(
        [
            A.Resize(height=256, width=256),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    train_dataset = AntsBeesDataset(root=os.path.join(data_dir, "train"))
    train_dataset.transforms = data_transforms

    test_dataset = AntsBeesDataset(root=os.path.join(data_dir, "val"))
    test_dataset.transforms = data_transforms

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    )
    num_ftrs = model.fc.in_features
    # We have only 2 classes
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    # Train model
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(len(train_loader))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(10):  # loop over the dataset multiple times
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels])
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print("Finished Training")

    # Save model
    torch.save(model.state_dict(), "model.pth")
