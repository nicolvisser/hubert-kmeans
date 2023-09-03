dependencies = ["torch"]

URLS = {
    "hubert-bshall": {
        "librispeech": {
            50: "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/kmeans-hubert-bshall-librispeech-50-7fc95527.pt",
            100: "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/kmeans-hubert-bshall-librispeech-100-15a551d8.pt",
            200: "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/kmeans-hubert-bshall-librispeech-200-de177bab.pt",
            500: "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/kmeans-hubert-bshall-librispeech-500-fd6f8c43.pt",
            1000: "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/kmeans-hubert-bshall-librispeech-1000-f063d3ed.pt",
            2000: "https://github.com/nicolvisser/hubert-kmeans/releases/download/v0.1/kmeans-hubert-bshall-librispeech-2000-1333c897.pt",
        }
    }
}

import torch

from kmeans import KMeansInference


def kmeans(
    features: str = "hubert-bshall",
    dataset: str = "librispeech",
    n_units: int = 500,
    pretrained: bool = True,
    progress: bool = True,
) -> KMeansInference:
    available_features = URLS.keys()
    assert (
        features in available_features
    ), f"features must be one of {available_features}"
    available_datasets = URLS[features].keys()
    assert (
        dataset in available_datasets
    ), f"dataset must be one of {available_datasets}, if you choose {features}"
    available_units = URLS[features][dataset].keys()
    assert (
        n_units in available_units
    ), f"n_units must be one of {available_units}, if you choose {features} and {dataset}"

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS[features][dataset][n_units],
            progress=progress,
            check_hash=True,
        )
        model = KMeansInference.load_from_checkpoint(checkpoint)
    else:
        raise NotImplementedError(
            "Only pretrained models are available. Set pretrained=True"
        )

    return model
