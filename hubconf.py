dependencies = ["torch"]

URLS = {
    50: "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/kmeans-50.pt",
    100: "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/kmeans-100.pt",
    200: "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/kmeans-200.pt",
    500: "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/kmeans-500.pt",
    1000: "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/kmeans-1000.pt",
    2000: "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/kmeans-2000.pt",
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
            URLS[n_clusters], progress=progress
        )
        model.cluster_centers = checkpoint["cluster_centers_"]
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
