from pathlib import Path
import numpy as np
from tqdm import tqdm
import click

# ================================================================================ #

mean_pooled_utterance_dir = Path(
    click.prompt(
        "Enter the path to the directory containing the mean pooled utterance features",
        type=click.Path(exists=True, file_okay=False),
    )
)

out_dir = Path(
    click.prompt(
        "Enter the path to the directory where the mean pooled speaker features will be saved",
        type=click.Path(exists=True, file_okay=False),
    )
)

# ================================================================================ #

subsets = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
]

for subset in subsets:
    print(f"Processing {subset}...")
    subset_dir = mean_pooled_utterance_dir / subset
    speaker_subdirs = sorted([d for d in subset_dir.iterdir() if d.is_dir()])
    for speaker_subdir in tqdm(speaker_subdirs):
        utterance_features_paths = speaker_subdir.rglob("*.npy")
        utterance_features = []
        for utterance_feature_path in utterance_features_paths:
            utterance_features.append(np.load(utterance_feature_path))
        mean_pooled_speaker_feature = np.mean(utterance_features, axis=0)
        out_path = out_dir / subset / speaker_subdir.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, mean_pooled_speaker_feature)
