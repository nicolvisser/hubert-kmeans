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


def kmeans(
    n_clusters: int,
    pretrained: bool = True,
    progress: bool = True,
) -> KMeansInference:
    assert n_clusters in URLS.keys(), f"n_clusters must be one of {URLS.keys()}"

    model = KMeansInference(k=n_clusters, d=768)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS[n_clusters], progress=progress
        )
        model.cluster_centers.data = torch.from_numpy(checkpoint["cluster_centers_"])
        model.eval()
    return model
