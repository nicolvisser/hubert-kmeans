# k-means models for HuBERT features

This repo contains more k-means models for the HuBERT model from https://github.com/bshall/hubert and https://github.com/bshall/soft-vc.

These k-means models can be used in conjunction with these [duration](https://github.com/nicolvisser/duration-predictor) and [acoustic](https://github.com/nicolvisser/acoustic-model) models.

## Available models

| features          | dataset         | n_units |
| ----------------- | --------------- | ------- |
| `"hubert-bshall"` | `"librispeech"` | `50`    |
|                   |                 | `100`   |
|                   |                 | `200`   |
|                   |                 | `500`   |
|                   |                 | `1000`  |
|                   |                 | `2000`  |

## Example Usage

```python
import torch, torchaudio

# load models
hubert = torch.hub.load(
    "bshall/hubert:main",
    "hubert_discrete",
    trust_repo=True,
).cuda()

kmeans = torch.hub.load(
    "nicolvisser/hubert-kmeans:main",
    "kmeans",
    features="hubert-bshall",
    dataset="librispeech",
    n_units=50, # <- change this to the number of units you want
    trust_repo=True,
).cuda()

# load audio
wav, sr = torchaudio.load("path/to/audio")
assert sr == 16000
wav = wav.unsqueeze(0).cuda()

# Extract HuBERT features
wav = torch.nn.functional.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
features, _ = hubert.encode(wav, layer=7)

# Cluster features
units = kmeans.predict(features.squeeze())

# Optionally dedupe the units (remove consequtive duplicates)
units = kmeans.predict_deduped(features.squeeze())

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
