from mlserver import MLModel
from mlserver.codecs import decode_args
import numpy as np
from PIL import Image
import timm


class TinyViTModel(MLModel):

    async def load(self) -> bool:
        self._model = timm.create_model("tiny_vit_5m_224.dist_in22k", pretrained=True)
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
        output = output.softmax(dim=1)
        return output.detach().numpy()
