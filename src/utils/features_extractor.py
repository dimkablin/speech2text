"""Feture extractor utilities"""
from io import BytesIO
from os import PathLike
import torch
import torchaudio
import librosa
import numpy as np


def stereo2mono(audio: torch.Tensor) -> torch.Tensor:
    """Convert audio from stereo to mono."""
    audio_mono = torch.mean(audio, dim=0)

    return audio_mono


def load_audio(path: str | PathLike) -> torch.Tensor:
    """ Load audio from file."""
    # load our wav file
    speech, sr = torchaudio.load(path)
    speech = stereo2mono(speech)
    resampler = torchaudio.transforms.Resample(sr, 16000)
    speech = resampler(speech)
    return speech.squeeze()

def split_audio(audio: torch.Tensor, sample_rate: int, chunk_size_sec=60) -> torch.Tensor:
    # вычисляем сколько чанков будет
    chunk_size = sample_rate * chunk_size_sec
    num_chunks = audio.size(0) // chunk_size
    chunks = []

    # проходимся по всем чанкам 
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = audio[start:end]
        chunks.append(chunk)

    # добавляем последний чанк
    last_chunk = audio[num_chunks * chunk_size:]
    # если у нас так вышло, что audio % (sample_rate * chunk_size_sec) == 0 
    # то скипаем последний чанк
    if len(last_chunk) > 0:
        chunks.append(last_chunk)

    return chunks

def split_audio2(audio: np.ndarray, sample_rate: int = 16000, top_db: int = 30):
    """ Split audio into segments based on silence using librosa. """
    intervals = librosa.effects.split(audio, top_db=top_db)
    print(intervals)
    segments = [audio[start:end] for start, end in intervals]
    return segments

def reduce_noise(audio: np.ndarray, sample_rate: int = 16000):
    """ Reduce noise from audio using a high-pass filter with librosa. """
    # Apply a high-pass filter to remove low-frequency noise
    audio_filtered = librosa.effects.preemphasis(audio)
    return audio_filtered