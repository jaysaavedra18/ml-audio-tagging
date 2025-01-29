import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_waveform(y: np.ndarray, sr: int) -> None:
    """Plot the waveform of an audio signal."""
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


def plot_spectogram(y: np.ndarray, sr: int) -> None:
    """Plot the spectrogram of an audio signal."""
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(
        librosa.power_to_db(S, ref=np.max),
        sr=sr,
        y_axis="mel",
        x_axis="time",
    )
    plt.colorbar(format="%+2.0d dB")
    plt.title("Mel Spectogram")
    plt.show()


def plot_loudness(rms: np.ndarray, sr: int) -> None:
    """Plot the loudness of an audio signal over time."""
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(rms[0], sr=sr)
    plt.title("Loudness (RMS) Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("RMS Energy")
    plt.show()
