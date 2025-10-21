import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torch import Tensor, device


imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)


tfms = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std
        ),
])


def fuse_embeddings(
        e_query: Tensor, e_cap: Tensor = None, e_img: Tensor = None,
        alpha: float = .0, beta: float = .0
) -> Tensor:
    """
    Fuse embeddings from query, caption (optional), and image (optional) based on weights alpha and beta.
    Ensures that the sum of alpha, beta, and the remaining weight equals 1.

    :param e_query: Embedding of the query.
    :param e_cap: Embedding of the caption (optional).
    :param e_img: Embedding of the image (optional).
    :param alpha: Weight for the caption embedding.
    :param beta: Weight for the image embedding.
    :return: Fused embedding.
    """
    # Ensure alpha and beta do not exceed 1 together
    if alpha + beta > 1.0:
        raise ValueError("The sum of alpha and beta should not exceed 1.")

    # Initialize the fused embedding with the query embedding
    fused_embedding = e_query.clone() * (1.0 - alpha - beta)

    # Add the caption embedding if provided
    if e_cap is not None and alpha > .0001:
        fused_embedding += alpha * e_cap

    # Add the image embedding if provided
    if e_img is not None and beta > .0001:
        fused_embedding += beta * e_img

    # Note: no need to normalize it since it's the weighted-average score b.t.w. given vector and each fused vector
    return fused_embedding


def dot_score1(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.

    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


def dot_score2(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.

    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return a @ b.T


def cosine2acc(cosine_similarity):
    return (cosine_similarity + 1) / 2