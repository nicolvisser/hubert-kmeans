import torch
from sklearn.cluster import KMeans


def load_model_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    n_clusters = checkpoint["cluster_centers_"].shape[0]

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.__dict__["n_features_in_"] = checkpoint["n_features_in_"]
    kmeans.__dict__["_n_threads"] = checkpoint["_n_threads"]
    kmeans.__dict__["cluster_centers_"] = checkpoint["cluster_centers_"]

    return kmeans

if __name__ == "__main__":
    kmeans = load_model_from_checkpoint("/home/nicolvisser/Workspace/pipeline/kmeans/checkpoints/units-50.pt")

    print(kmeans.cluster_centers_.shape)