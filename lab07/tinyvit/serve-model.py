import os
import json
import timm
import torch
import numpy as np

from PIL import Image

from mlserver import MLModel
from mlserver.codecs import decode_args

class TinyViTModel(MLModel):

    def __init__(self, *args, **kwargs):
        super(TinyViTModel, self).__init__(*args, **kwargs)
        dirname = os.path.dirname(__file__)
        with open(f'{dirname}/imagenet-22k-id2label.json') as f:
            d = json.load(f)
        self.label_dict = d

    async def load(self) -> bool:
        self._model = timm.create_model('tiny_vit_5m_224.dist_in22k', pretrained=True)
        self._model.eval()

        data_config = timm.data.resolve_data_config({}, model=self._model)
        self._transforms = timm.data.create_transform(**data_config, is_training=False)
        self.ready = True
        return self.ready

    @decode_args
    async def predict(self, payload: np.ndarray) -> np.ndarray:
        img = Image.fromarray(payload)
        img = self._transforms(img).unsqueeze(0)
        output = self._model(img)
        _, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
        #top5_classes = [self.label_dict[str(idx)] for idx in top5_class_indices.squeeze().tolist()]
        top5_classes = [
            self.label_dict[str(idx)].split(',')[0].strip()
            for idx in top5_class_indices.squeeze().tolist()
        ]
        return np.array(top5_classes)
    
        #output = output.softmax(dim=1)
        #return output.detach().numpy()