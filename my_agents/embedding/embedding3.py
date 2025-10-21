# Written by Mingxuan Liu
# Model for OpenAI embedding3 model
import openai
import os
from openai import OpenAI
from my_agents.embedding.utils import *
import torch

_API_BASE = {
    "text-embedding-3-small": "https://api.openai.com/v1/embeddings",
    "text-embedding-3-large": "https://api.openai.com/v1/embeddings",
}

_MODEL_ZOO = {
    "emb3_small": "text-embedding-3-small",
    "emb3_large": "text-embedding-3-large",
}


class Embedding3Model:
    def __init__(self, model_size, device='cuda'):
        if model_size not in _MODEL_ZOO:
            raise ValueError("{} model does not exit".format(model_size))

        self.model_size = _MODEL_ZOO[model_size]
        self.device = device
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def get_embedding_dim(self):
        return 1536 if self.model_size == "text-embedding-3-small" else 3072

    def encode_text(self, in_text):
        text = [t.replace("\n", " ") for t in in_text]
        total_embeddings = []
        for t in text:
            total_embeddings.append(
                torch.tensor(self.client.embeddings.create(input=[t], model=self.model_size).data[0].embedding).to(self.device)
            )
        return torch.stack(total_embeddings, dim=0).to(self.device)

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
