dependencies = ["torch"]

URLS = {
    "kmeans_50": "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/units-50.pt",
    "kmeans_100": "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/units-100.pt",
    "kmeans_200": "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/units-200.pt",
    "kmeans_500": "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/units-500.pt",
    "kmeans_1000": "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/units-1000.pt",
    "kmeans_2000": "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/units-2000.pt",
}

import torch

from kmeans import KMeansInference


def _kmeans(
    n_clusters: int,
    pretrained: bool = True,
    progress: bool = True,
) -> KMeansInference:
    model = KMeansInference(k=n_clusters, d=768)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS[f"kmeans_{n_clusters}"], progress=progress
        )
        model = KMeansInference.load_from_checkpoint(checkpoint)
        model.eval()
    return model


def kmeans_50(
    pretrained: bool = True,
    progress: bool = True,
) -> KMeansInference:
    """
    KMeans model with 50 clusters trained on the HuBERT embeddings of the LibriSpeech train-clean-100 dataset.
    Args:
        pretrained (bool): If True, returns a model pre-trained on LibriSpeech train-clean-100
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _kmeans(50, pretrained, progress)


def kmeans_100(
    pretrained: bool = True,
    progress: bool = True,
) -> KMeansInference:
    """
    KMeans model with 100 clusters trained on the HuBERT embeddings of the LibriSpeech train-clean-100 dataset.
    Args:
        pretrained (bool): If True, returns a model pre-trained on LibriSpeech train-clean-100
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _kmeans(100, pretrained, progress)


def kmeans_200(
    pretrained: bool = True,
    progress: bool = True,
) -> KMeansInference:
    """
    KMeans model with 200 clusters trained on the HuBERT embeddings of the LibriSpeech train-clean-100 dataset.
    Args:
        pretrained (bool): If True, returns a model pre-trained on LibriSpeech train-clean-100
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _kmeans(200, pretrained, progress)


def kmeans_500(
    pretrained: bool = True,
    progress: bool = True,
) -> KMeansInference:
    """
    KMeans model with 500 clusters trained on the HuBERT embeddings of the LibriSpeech train-clean-100 dataset.
    Args:
        pretrained (bool): If True, returns a model pre-trained on LibriSpeech train-clean-100
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _kmeans(500, pretrained, progress)


def kmeans_1000(
    pretrained: bool = True,
    progress: bool = True,
) -> KMeansInference:
    """
    KMeans model with 1000 clusters trained on the HuBERT embeddings of the LibriSpeech train-clean-100 dataset.
    Args:
        pretrained (bool): If True, returns a model pre-trained on LibriSpeech train-clean-100
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _kmeans(1000, pretrained, progress)


def kmeans_2000(
    pretrained: bool = True,
    progress: bool = True,
) -> KMeansInference:
    """
    KMeans model with 2000 clusters trained on the HuBERT embeddings of the LibriSpeech train-clean-100 dataset.
    Args:
        pretrained (bool): If True, returns a model pre-trained on LibriSpeech train-clean-100
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _kmeans(2000, pretrained, progress)
