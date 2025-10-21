# Written by Mingxuan Liu

import clip
from my_agents.embedding.utils import *

_MODEL_ZOO = ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336"]


class CLIP:
    def __init__(
            self,
            model_size,
            device='cuda',
            jit=False,
            parallel=False,
    ):
        if model_size not in _MODEL_ZOO:
            raise ValueError("CLIP {} is not a valid".format(model_size))

        self.model_size = model_size
        self.device = device
        self.jit = jit
        self.parallel = parallel
        self.modalities = ["vision", "language"]
        self.encoder = None
        self.preprocesser = None

        self.__create_model()

    def __create_model(self):
        self.encoder, self.preprocesser = clip.load(self.model_size, device=self.device, jit=self.jit)
        self.encoder.eval()
        self.encoder.requires_grad_(False)

        # if parallelize
        if self.parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.encoder = self.encoder.to(self.device)

    def encode_text(self, input, truncate=False):
        tokens = clip.tokenize(input, truncate=truncate).to(self.device)
        if self.parallel:
            embedding = self.encoder.module.encode_text(tokens)
        else:
            embedding = self.encoder.encode_text(tokens)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def encode_image(self, input):
        if self.parallel:
            embedding = self.encoder.module.encode_image(input)
        else:
            embedding = self.encoder.encode_image(input)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def compute_similarity(self, queries, passages, mod_query='language', mod_passage='language'):
        """
        Computes the dot-product dot_prod(a[i], b[j]) for all i and j.

        :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
        """
        if mod_query not in self.modalities or mod_passage not in self.modalities:
            raise ValueError(f"{mod_query} or {mod_passage} is not supported!")
        embedding1 = self.encode_image(queries) if mod_query == "vision" else self.encode_text(queries)
        embedding2 = self.encode_image(passages) if mod_passage == "vision" else self.encode_text(passages)
        similarity = dot_score2(embedding1, embedding2)
        return similarity


if __name__ == "__main__":
    pass
