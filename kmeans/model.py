import torch
import torch.nn as nn


class KMeansInference(nn.Module):
    def __init__(self, k, d):
        super().__init__()
        self.cluster_centers = nn.Parameter(torch.zeros(k, d))

    def forward(self, features):
        distances = torch.cdist(features, self.cluster_centers)
        indices = torch.argmin(distances, dim=1)
        return indices

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        cluster_centers = torch.from_numpy(checkpoint["cluster_centers_"])
        model = cls(cluster_centers.shape[0], cluster_centers.shape[1])
        model.cluster_centers.data = cluster_centers
        return model
