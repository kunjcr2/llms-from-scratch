# Audio Encoders: A First Learning Project

This folder is a small, practical introduction to audio encoders and audio classification. The accompanying notebook trains a model to recognize ten spoken commands from the public Google Speech Commands dataset.

## What an audio encoder does

Audio is a sequence of numbers. At a 16,000 Hz sample rate, one second of mono audio is 16,000 amplitude values. A raw waveform is useful for playback, but it is not yet a compact description that a classifier can use easily.

An encoder transforms that long waveform into a shorter feature representation. The classifier uses that representation to choose a label.

The flow in this project is:

audio waveform -> convolutional encoder -> fixed-size embedding -> linear classifier -> command label

For example, an input saying “yes” becomes a tensor with shape `[1, 16000]`. The encoder produces many learned feature maps over time, then pools them into one vector. That vector is the audio embedding. The final classifier scores every possible command and picks the highest score.

## A normal first-level audio encoder

The first encoder layer is often a one-dimensional convolution over the waveform.

It has a small filter that slides from left to right across the audio samples. At each position, it combines nearby samples into one value. Each filter can learn to react to a simple local sound pattern, such as a sharp change, a short periodic vibration, or the beginning of a phoneme.

This first layer is deliberately local. It does not understand an entire word by itself. Its job is to turn raw samples into useful low-level features. Later layers combine those features over longer spans of time, so the model can recognize word-level patterns.

In the notebook, the first convolution uses a kernel wider than one sample and a stride greater than one. The wider kernel lets it inspect a short audio window. The stride reduces the time dimension, so later layers process fewer positions. ReLU adds nonlinearity, and pooling reduces the sequence further while keeping strong activations.

## Why this is called an encoder

The encoder changes the representation, but does not make the final decision. Its output is an embedding: a compact learned summary of the clip. The classifier is the final linear layer that turns the embedding into one score per command.

Keeping those jobs separate is useful. After learning an encoder, you can reuse its embeddings for another task, such as a new set of commands or a small similarity search experiment.

## Dataset and labels

The notebook downloads Speech Commands v0.02 through torchaudio on its first run. It uses a ten-command subset:

- yes
- no
- up
- down
- left
- right
- on
- off
- stop
- go

Speech Commands provides official validation and test file lists. The notebook respects those lists, so the model is evaluated on recordings it did not train on.

## Run the notebook

Open `train_simple_audio_classifier.ipynb` from this directory. It assumes a Python environment with PyTorch, torchaudio, matplotlib, and tqdm installed. The first data-loading cell downloads the dataset into `audio_encoders/data`, which is ignored by Git. Start with the default two epochs to verify everything works, then increase the epoch count to learn more from the results.

The model is intentionally small and simple. It is designed to make the path from waveform to embedding to prediction easy to inspect, not to set a benchmark.
