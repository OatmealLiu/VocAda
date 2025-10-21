import spacy
from my_agents.embedding.utils import *
import torch

_MODEL_ZOO = {
    "small": "en_core_web_sm",
    "middle": "en_core_web_md",
    "large": "en_core_web_lg",
    "base": "en_core_web_trf",
}


class SpacyEmbedding:
    def __init__(self, model_name, device='cuda'):
        if model_name not in _MODEL_ZOO:
            raise ValueError("{} is not a valid".format(model_name))
        self.model_size = _MODEL_ZOO[model_name]
        self.device = device
        self.nlp = spacy.load(self.model_size)

    def encode_text(self, input):
        total_embeddings = []
        for text in input:
            tokens = self.nlp(text)
            embeddings = [tkn.vector_norm for tkn in tokens]
            embeddings = torch.tensor(embeddings).to(self.device)
            embeddings = F.normalize(embeddings.mean(dim=0))
            total_embeddings.extend(embeddings)
        total_embeddings = torch.tensor(total_embeddings).to(self.device)
        total_embeddings = F.normalize(total_embeddings.mean(dim=0))
        return total_embeddings

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
