import requests
import numpy as np

from PIL import Image

from mlserver.codecs import NumpyCodec
from mlserver.types import InferenceRequest

inference_url = 'http://localhost:8080/v2/models/tinyvit/infer'
input_data = np.array(Image.open('../imgs/cat.jpg'))

inference_request = InferenceRequest(
    inputs=[NumpyCodec.encode_input(name='payload', payload=input_data)]
)

res = requests.post(inference_url, json=inference_request.dict())
print(res.json()['outputs'][0]['data'])