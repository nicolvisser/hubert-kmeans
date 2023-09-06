import torch, torchaudio
from pathlib import Path

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
    n_units=50,
    trust_repo=True,
).cuda()

# load audio
librispeech_dir = Path("./data/LibriSpeech")
dataset = torchaudio.datasets.LIBRISPEECH(
    root=librispeech_dir.parent,
    url="dev-clean",
    folder_in_archive=librispeech_dir.name,
    download=True,
)
wav, sr, *_ = dataset[0]
assert sr == 16000
wav = wav.unsqueeze(0).cuda()

# Extract HuBERT features
wav = torch.nn.functional.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
features, _ = hubert.encode(wav, layer=7)

# Cluster features
units = kmeans.predict(features.squeeze())

# Optionally dedupe the units (remove consequtive duplicates)
units = kmeans.predict_deduped(features.squeeze())

print(units)
