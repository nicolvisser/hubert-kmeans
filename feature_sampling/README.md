This folder contains all the required steps to reproduce the feature set used to cluster hubert features.

The set of features is the hubert features from all utterances from 10 diverse speakers from the train-100-clean subset of LibriSpeech.

1.  Extract mean pooled feature from each utterance in LibriSpeech with `mean_pool_utterances.py`
2.  Calculate the mean (of means) for each speaker `mean_pool_speakers.py`
3.  Select the 10 most diverse speakers `select_speakers.py`
4.  Extract and combine the features for each utterance from the selected speakers `extract_features.py`
