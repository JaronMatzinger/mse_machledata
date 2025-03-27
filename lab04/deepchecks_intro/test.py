import albumentations as A
from albumentations.pytorch import ToTensorV2
from deepchecks.vision.vision_data import BatchOutputFormat, VisionData
from deepchecks.vision.suites import model_evaluation
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

import os

from data import AntsBeesDataset


def build_collate_fn(model, device) -> callable:
    def _collate_fn(batch) -> BatchOutputFormat:
        """Return a batch of images, labels and predictions for a batch of data. The expected format is a dictionary with
        the following keys: 'images', 'labels' and 'predictions', each value is in the deepchecks format for the task.
        You can also use the BatchOutputFormat class to create the output.
        """
        # batch received as iterable of tuples of (image, label) and transformed to tuple of iterables of images and labels:
        batch = tuple(zip(*batch))

        # images:
        inp = torch.stack(batch[0]).detach().numpy().transpose((0, 2, 3, 1))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        inp = std * inp + mean
        images = np.clip(inp, 0, 1) * 255

        # labels:
        labels = batch[1]

        # predictions:
        logits = model.to(device)(torch.stack(batch[0]).to(device))
        predictions = nn.Softmax(dim=1)(logits)
        return BatchOutputFormat(images=images, labels=labels, predictions=predictions)

    return _collate_fn


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load data
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

    train_dataset = AntsBeesDataset(root=os.path.join(data_dir, "val"))
    train_dataset.transforms = data_transforms

    test_dataset = AntsBeesDataset(root=os.path.join(data_dir, "val"))
    test_dataset.transforms = data_transforms

    # Load model from 'model.pth' file
    model = torchvision.models.resnet18()
    num_ftrs = model.fc.in_features
    # We have only 2 classes
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    model.to(device)

    collate_fn = build_collate_fn(model, device)
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
    )

    train_data = VisionData(train_loader, task_type="classification")
    test_data = VisionData(test_loader, task_type="classification")

    suite = model_evaluation()
    result = suite.run(
        train_dataset=train_data, test_dataset=test_data, max_samples=5000
    )
    result.save_as_html("report.html", as_widget=False)
