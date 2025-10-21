# Written by Mingxuan Liu
import torch


_DINOV2_ZOO = [
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
]

_DINOV1_ZOO =[
    "dino_vits16",
    "dino_vits8",
    "dino_vitb16",
    "dino_vitb8",
]


class DINO:
    def __init__(
            self,
            model_name,
            device='cuda',
            parallel=False
    ):
        self.model_name = model_name
        self.device = device
        self.parallel = parallel

        self.encoder = None
        self.preprocesser = None

        self.__create_model()

    def __create_model(self):
        if self.model_name in _DINOV1_ZOO:
            self.encoder = torch.hub.load('facebookresearch/dino:main', self.model_name)
        elif self.model_name in _DINOV2_ZOO:
            self.encoder = torch.hub.load('facebookresearch/dinov2', self.model_name)
        else:
            raise ValueError
        self.encoder.eval()
        self.encoder = self.encoder.to(self.device)

    def encode_image(self, input):
        return self.encoder(input)


if __name__ == "__main__":
    pass

