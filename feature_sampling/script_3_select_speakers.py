from pathlib import Path
import numpy as np
import click

# ================================================================================ #


def farthest_point_sampling(data, K):
    """
    Select K datapoints using the Farthest Point Sampling method.

    Parameters:
    - data: np.array of shape (M, d)
    - K: int, number of datapoints to select

    Returns:
    - subset: np.array of shape (K, d)
    """

    # Randomly select the first data point
    first_index = np.random.randint(data.shape[0])
    subset_indices = [first_index]

    while len(subset_indices) < K:
        # Calculate distances from the latest selected point to all points
        distances = np.linalg.norm(data - data[subset_indices[-1]], axis=1)

        # Update the farthest distance for each point based on previously selected points
        for i in subset_indices[:-1]:
            distances = np.minimum(distances, np.linalg.norm(data - data[i], axis=1))

        # Select the point with the maximum distance
        subset_indices.append(np.argmax(distances))

    subset = data[subset_indices]

    return subset, subset_indices


# ================================================================================ #

feature_dir = Path(
    click.prompt(
        "Enter the path to the directory containing the mean pooled speaker features",
        type=click.Path(exists=True, file_okay=False),
    )
)

num_speakers = click.prompt("Enter the number of speakers to sample from", type=int)

speaker_ids_file_path = Path(
    click.prompt(
        "Enter the path to the file to save the selected speaker IDs to",
        type=click.Path(exists=False, dir_okay=False),
    )
).with_suffix(".npy")

# ================================================================================ #

feature_paths = sorted(list(feature_dir.rglob("*.npy")))
speaker_ids = [feature_path.stem for feature_path in feature_paths]

speaker_features = []
for feature_path in feature_paths:
    speaker_features.append(np.load(feature_path))
speaker_features = np.array(speaker_features)

np.random.seed(3459)
features_subset, subset_indices = farthest_point_sampling(
    speaker_features,
    num_speakers,
)
selected_speaker_ids = [speaker_ids[i] for i in subset_indices]
np.save(speaker_ids_file_path, selected_speaker_ids)
