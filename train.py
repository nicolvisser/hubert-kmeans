import click
import faiss
import numpy as np
import torch

feature_data_path = click.prompt(
    "Enter the path to the feature data file", type=click.Path(exists=True)
)

k = click.prompt("Enter the number of clusters", type=int)

feature_data = np.load(feature_data_path)

d = feature_data.shape[1]

kmeans = faiss.Kmeans(
    d,
    k,
    niter=100,
    nredo=10,
    max_points_per_centroid=50000,
    spherical=False,
    verbose=True,
)

kmeans.train(feature_data)

kmeans_checkpoint = {}
kmeans_checkpoint["cluster_centers_"] = kmeans.centroids
kmeans_checkpoint["_n_threads"] = 4
kmeans_checkpoint["n_features_in_"] = kmeans.centroids.shape[1]

torch.save(
    kmeans_checkpoint,
    f"checkpoints/kmeans-{k}.pt",
)
