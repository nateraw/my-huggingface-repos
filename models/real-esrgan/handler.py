from base64 import b64encode
from io import BytesIO
from pathlib import Path

import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
from realesrgan import RealESRGANer


class EndpointHandler:
    def __init__(self, path=""):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=str(Path(path) / "RealESRGAN_x4plus.pth"),
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
        )

    def __call__(self, data):
        """
        Args:
            data (:obj:):
                includes the input data and the parameters for the inference.
        Return:
            A :obj:`dict`:. base64 encoded image
        """
        image = data.pop("inputs", data)
        # image = Image.open(BytesIO(image)).convert("RGB")
        image = np.array(image)
        image = image[:, :, ::-1]  # RGB -> BGR

        image, _ = self.upsampler.enhance(image, outscale=4)
        image = image[:, :, ::-1]  # BGR -> RGB
        image = Image.fromarray(image)

        # encode image as base 64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = b64encode(buffered.getvalue())

        # postprocess the prediction
        return {"image": img_str.decode()}