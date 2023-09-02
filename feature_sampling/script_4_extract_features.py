import numpy as np
from pathlib import Path
from tqdm import tqdm
import click

# ================================================================================ #

selected_speaker_ids_path = Path(
    click.prompt(
        "Enter the path to the file containing the selected speaker IDs",
        type=click.Path(exists=True, dir_okay=False),
    )
)

feature_dir = Path(
    click.prompt(
        "Enter the path to the directory containing the hubert features",
        type=click.Path(exists=True, file_okay=False),
    )
)

# ================================================================================ #

speaker_ids = np.load(selected_speaker_ids_path)

features = []
for speaker_id in tqdm(speaker_ids):
    utterance_paths = feature_dir.rglob(f"{speaker_id}-*.npy")
    for utterance_path in utterance_paths:
        features.append(np.load(utterance_path))
features = np.vstack(features)

np.save("features-50-from-train-clean-100.npy", features)
