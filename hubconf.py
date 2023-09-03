dependencies = ["torch"]

URLS = {
    50: "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/kmeans-50-7fc95527.pt",
    100: "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/kmeans-100-15a551d8.pt",
    200: "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/kmeans-200-de177bab.pt",
    500: "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/kmeans-500-fd6f8c43.pt",
    1000: "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/kmeans-1000-f063d3ed.pt",
    2000: "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/kmeans-2000-1333c897.pt",
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
            URLS[n_clusters],
            progress=progress,
            check_hash=True,
        )
        model.cluster_centers.data = torch.from_numpy(checkpoint["cluster_centers_"])
        model.eval()
    return model
