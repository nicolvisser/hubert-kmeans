import torch
import torch.nn as nn


class KMeansInference(nn.Module):
    def __init__(self, k, d):
        super().__init__()
        self.cluster_centers = nn.Parameter(torch.zeros(k, d))

    def forward(self, features):
        distances = torch.cdist(features, self.cluster_centers)
        indices = torch.argmin(distances, dim=-1)
        return indices

    @torch.inference_mode()
    def predict(self, features):
        assert (
            features.ndim == 2
        ), f"Expected 2D tensor during inference, got {features.ndim}"
        return self.forward(features)

    @torch.inference_mode()
    def predict_deduped(self, features):
        indices = self.predict(features)
        diffs = torch.cat(
            (
                indices[:-1] != indices[1:],
                torch.tensor([True], dtype=torch.bool, device=indices.device),
            )
        )
        return indices[diffs]

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        cluster_centers = torch.from_numpy(checkpoint["cluster_centers_"])
        model = cls(cluster_centers.shape[0], cluster_centers.shape[1])
        model.cluster_centers.data = cluster_centers
        return model
