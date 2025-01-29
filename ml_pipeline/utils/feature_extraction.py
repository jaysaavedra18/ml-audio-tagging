import librosa
import librosa.display
import numpy as np

debug = False


def analyze_audio(audio_path: str) -> np.ndarray:
    """Analyze an audio file and extract its properties."""
    # Load an audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Extract tempo and bpm
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    # RMS energy is a measure of the audio signal's strength, calculate the avg loudness.
    loudness = librosa.feature.rms(y=y).mean()

    # Onset detection shows when musical events occur in the audio signal.
    avg_interval, std_interval = analyze_onset_intervals(y, sr)

    # Zero crossing rate is the rate of sign changes along a signal.
    zcr = librosa.feature.zero_crossing_rate(y=y).mean()

    # Spectral features capture the frequency content of the audio signal.
    (
        spectral_rolloff,
        spectral_flatness,
        spectral_contrast,
        spectral_bandwidth,
        spectral_centroid,
    ) = analyze_spectral_features(y=y, sr=sr)

    # MFCCs capture the song's timbre and spectral texture, summarizing its tonal qualities over time.
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Chroma features capture the harmonic content of a song, indicating its musical key.
    chroma, chroma_cens, chroma_cqt, chroma_stft = analyze_chroma_features(y, sr)

    # Harmonic-to-Noise Ratio (HNR) measures the ratio of harmonic content to noise. (Music vs. Speech)
    harmonics = librosa.effects.harmonic(y=y)
    hnr = np.mean(harmonics) / np.mean(y)

    # Tonnetz features capture tonal content in music.
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

    # Build the feature vector by flattening the analysis results
    features = np.concatenate(
        [
            tempo,  # BPM (1) (0)
            [loudness],  # Avg loudness (1) (1)
            [avg_interval, std_interval],  # Onset intervals (2) (2)
            [np.mean(zcr)],  # Zero crossing rate (1) (3)
            [np.mean(spectral_centroid)],  # Spectral centroid (1) (4)
            [np.mean(spectral_bandwidth)],  # Spectral bandwidth (1) (5)
            [np.mean(spectral_rolloff)],  # Spectral rolloff (1) (6)
            [np.mean(spectral_flatness)],  # Spectral flatness (1) (7)
            np.mean(spectral_contrast, axis=1),  # Spectral contrast (6) (8)
            np.mean(mfccs, axis=1),  # MFCCs (13) (9)
            np.mean(chroma, axis=1),  # Chroma (12) (10)
            np.mean(chroma_cens, axis=1),  # Chroma CENS (12) (11)
            np.mean(chroma_cqt, axis=1),  # Chroma CQT (12) (12)
            np.mean(chroma_stft, axis=1),  # Chroma STFT (12) (13)
            [hnr],  # HNR (1) (14)
            np.mean(tonnetz, axis=1),  # Tonnetz (6) (15)
        ],
    )

    # Print the extracted features if debug mode is enabled
    if debug:
        print_features(features)

    return features


def analyze_onset_intervals(y: np.ndarray, sr: int) -> tuple:
    """Analyze the intervals between musical onsets in an audio signal."""
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    intervals = np.diff(onset_times)
    avg_interval = np.mean(intervals) if len(intervals) > 0 else 0
    std_interval = np.std(intervals) if len(intervals) > 0 else 0
    return avg_interval, std_interval


def analyze_chroma_features(y: np.ndarray, sr: int) -> tuple:
    """Analyze the chroma features of an audio signal."""
    # Chroma features capture the harmonic content of the audio signal.
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # Chroma cens captures the tonal content of the audio signal.
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    # Chroma cqt captures the tonal content of the audio signal.
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    # Chroma stft represents the energy content of different pitches in the audio signal.
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    return chroma, chroma_cens, chroma_cqt, chroma_stft


def analyze_spectral_features(y: np.ndarray, sr: int) -> tuple:
    """Analyze the spectral features of an audio signal."""
    # Spectral rolloff distinguishes between harmonic and percussive components of a signal.
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    # Spectral flatness measures the noisiness of a signal.
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    # Spectral contrast measures the difference in amplitude between peaks and valleys in a signal.
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    # Spectral bandwidth measures the width of the frequency range in which a signal's energy is concentrated.
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    # Spectral centroid indicates the "center of mass" of the frequency distribution.
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return (
        spectral_rolloff,
        spectral_flatness,
        spectral_contrast,
        spectral_bandwidth,
        spectral_centroid,
    )


def print_features(features: np.array) -> None:
    """Print extracted audio features in a clean format."""
    print("Extracted Features:\n")

    # Print each feature, extracting values from arrays where necessary
    print(f"Tempo: {features[0]}\n")
    print(f"Loudness: {features[1]}\n")
    print(f"Onset Intervals (Avg, Std): {features[2]}, {features[3]}\n")
    print(f"Zero Crossing Rate: {features[4]}\n")
    print(f"Spectral Centroid: {features[5]}\n")
    print(f"Spectral Bandwidth: {features[6]}\n")
    print(f"Spectral Rolloff: {features[7]}\n")
    print(f"Spectral Flatness: {features[8]}\n")
    print(f"Spectral Contrast: {features[9:16]}\n")
    print(f"MFCCs: {features[16:29]}\n")
    print(f"Chroma: {features[29:41]}\n")
    print(f"Chroma CENS: {features[41:53]}\n")
    print(f"Chroma CQT: {features[53:65]}\n")
    print(f"Chroma STFT: {features[65:77]}\n")
    print(f"HNR: {features[77]}\n")
    print(f"Tonnetz: {features[78:]}\n")


debug = True
features = analyze_audio("/Users/saavedj/Downloads/music/misc/20 Min.mp3")
