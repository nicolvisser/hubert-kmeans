from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

# ================================================================================ #

feature_input_directory = Path(
    click.prompt(
        "Enter the path to the directory containing the hubert features",
        type=click.Path(exists=True, file_okay=False, dir_okay=True),
    )
)

mean_output_directory = Path(
    click.prompt(
        "Enter the path to the directory where the mean-pooled features will be saved",
        type=click.Path(exists=False, file_okay=False, dir_okay=True),
    )
)

# ================================================================================ #

feature_paths = list(feature_input_directory.rglob("*.npy"))

for feature_path in tqdm(feature_paths):
    relative_path = feature_path.relative_to(feature_input_directory)
    mean_output_path = mean_output_directory / relative_path

    mean_output_path.parent.mkdir(parents=True, exist_ok=True)

    feature = np.load(feature_path)
    mean_feature = np.mean(feature, axis=0)

    np.save(mean_output_path, mean_feature)
