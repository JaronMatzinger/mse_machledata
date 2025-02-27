import numpy as np
import PIL
import torchvision
import urllib.request

import os
import zipfile


class AntsBeesDataset(torchvision.datasets.ImageFolder):
    """Custom dataset class that inherits from torchvision.datasets.ImageFolder
    and overrides __getitem__ method to be compatible with albumentations.
    """

    def __getitem__(self, index: int):
        """overrides __getitem__ to be compatible to albumentations"""
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = self.get_cv2_image(sample)
        if self.transforms is not None:
            transformed = self.transforms(image=sample, target=target)
            sample, target = transformed["image"], transformed["target"]
        else:
            if self.transform is not None:
                sample = self.transform(image=sample)["image"]
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target

    def get_cv2_image(self, image):
        """Converts PIL image to CV2 image"""
        if isinstance(image, PIL.Image.Image):
            return np.array(image).astype("uint8")
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise RuntimeError("Only PIL.Image and CV2 loaders currently supported!")


if __name__ == "__main__":
    url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
    urllib.request.urlretrieve(url, "./hymenoptera_data.zip")

    with zipfile.ZipFile("./hymenoptera_data.zip", "r") as zip_ref:
        zip_ref.extractall(".")
