# Written by Mingxuan Liu

import torch
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer
from my_agents.embedding.utils import *


_MODEL_ZOO = {
    # "sbert_mini": "all-MiniLM-L6-v2",
    "sbert_mini": "multi-qa-MiniLM-L6-cos-v1",
    "sbert_base": "all-mpnet-base-v2",
    "sbert_search": "multi-qa-mpnet-base-dot-v1",
}


class SBERT:
    def __init__(self, model_size, device='cuda'):
        if model_size not in _MODEL_ZOO:
            raise ValueError("{} is not a valid".format(model_size))

        self.model_size = _MODEL_ZOO[model_size]
        self.device = device
        self.encoder = None
        self.__create_sbert()

    def __create_sbert(self):
        self.encoder = SentenceTransformer(self.model_size, device=self.device)

    # def encode_text(self, input):
    #     embedding = self.encoder(input, convert_to_tensor=True).to(self.device)
    #     embedding = F.normalize(embedding, p=2, dim=1)
    #     return embedding

    def encode_text(self, input):
        embedding = self.encoder.encode(input)
        embedding = torch.tensor(embedding)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.to(self.device)

    def compute_similarity(self, queries, passages):
        """
        Computes the dot-product dot_prod(a[i], b[j]) for all i and j.

        :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
        """
        query_embedding = self.encode_text(queries)
        passage_embedding = self.encode_text(passages)

        similarity = dot_score1(query_embedding, passage_embedding)
        return similarity


if __name__ == "__main__":
    pass






