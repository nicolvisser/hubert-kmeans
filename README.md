# k-means models for hubert features

This repo contains more k-means models for the hubert-discrete model from https://github.com/bshall/hubert

## Available models

| # Units | Model        |
| ------- | ------------ |
| 50      | `units_50`   |
| 100     | `units_100`  |
| 200     | `units_200`  |
| 500     | `units_500`  |
| 1000    | `units_1000` |
| 2000    | `units_2000` |

## Usage

```python
import torch, torchaudio

# Load hubert checkpoint
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).cuda()

# Load audio
wav, sr = torchaudio.load("path/to/wav")
assert sr == 16000
wav = wav.unsqueeze(0).cuda()

# Extract features
wav = torch.nn.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
features, _ = hubert.encode(wav, layer=7)

# Load k-means model
kmeans = torch.hub.load("nicolvisser/hubert_kmeans:main", "units_50", trust_repo=True).cuda()

# Cluster features
units = kmeans(features)


```

## Training Information

The models were trained by using features from all utterances from 10 diverse speakers from the LibriSpeech train-clean-100 subset.

Speaker IDs: `['7226', '1447', '412', '2989', '5778', '78', '7794', '6563', '458', '26']`

Rough scripts used for speaker selection can be found in `feature_sampling/`

[Faiss](https://github.com/facebookresearch/faiss) was used to train the models with parameters:

```python
niter=100,
nredo=10,
max_points_per_centroid=50000,
```
