import librosa
import numpy as np
import os
from pathlib import Path

def load_audio(file_path):
    """Load audio file and return audio time series and sampling rate."""
    if not Path(file_path).exists():
        print(f"Error: File {file_path} does not exist.")
        return None, None
    try:
        audio, sr = librosa.load(file_path, sr=None)  # Use native sampling rate
        print(f"Loaded {file_path} - Sampling Rate: {sr} Hz, Duration: {len(audio)/sr:.2f} seconds")
        return audio, sr
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None

def extract_features(audio, sr):
    """Extract audio features for comparison."""
    try:
        # MFCCs (timbre)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Tempo (rhythm)
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        tempo = float(tempo) if isinstance(tempo, np.ndarray) else tempo
        
        # Spectral Centroid (brightness/tone)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        centroid_mean = np.mean(spectral_centroid)
        
        features = {
            'mfcc': mfcc_mean,
            'tempo': tempo,
            'spectral_centroid': centroid_mean
        }
        print(f"Features extracted - Tempo: {tempo:.2f} BPM, Centroid: {centroid_mean:.2f} Hz")
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def compare_songs(file1, file2, mfcc_threshold=0.95, tempo_threshold=5, centroid_threshold=200):
    """
    Compare two audio files based on MFCC, tempo, and spectral centroid.
    Returns True if similar, False if different.
    """
    print(f"\nComparing {file1} and {file2}...")

    # Load audio files
    audio1, sr1 = load_audio(file1)
    audio2, sr2 = load_audio(file2)
    
    if audio1 is None or audio2 is None:
        print("Comparison failed due to loading errors.")
        return False
    
    # Extract features
    features1 = extract_features(audio1, sr1)
    features2 = extract_features(audio2, sr2)
    
    if features1 is None or features2 is None:
        print("Comparison failed due to feature extraction errors.")
        return False
    
    # Calculate similarities/differences
    try:
        # MFCC cosine similarity (timbre similarity)
        mfcc_similarity = np.dot(features1['mfcc'], features2['mfcc']) / (
            np.linalg.norm(features1['mfcc']) * np.linalg.norm(features2['mfcc'])
        )
        # Tempo difference
        tempo_diff = abs(features1['tempo'] - features2['tempo'])
        # Spectral centroid difference
        centroid_diff = abs(features1['spectral_centroid'] - features2['spectral_centroid'])
        
        # Debugging output
        print(f"MFCC Similarity: {mfcc_similarity:.3f} (Threshold: {mfcc_threshold})")
        print(f"Tempo Difference: {tempo_diff:.2f} BPM (Threshold: {tempo_threshold})")
        print(f"Centroid Difference: {centroid_diff:.2f} Hz (Threshold: {centroid_threshold})")
        
        # Determine similarity
        is_similar = (
            mfcc_similarity >= mfcc_threshold and
            tempo_diff <= tempo_threshold and
            centroid_diff <= centroid_threshold
        )
        
        print(f"Result: {'Similar' if is_similar else 'Different'}")
        return is_similar
    except Exception as e:
        print(f"Error during comparison: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Define file paths
    song1_path = "./MSE25_DSP_data/4ze23ae2yv.mp3"
    song2_path = "./MSE25_DSP_data/ujqpakwgzp.mp3"
    song3_path = "./MSE25_DSP_data/RungLaThap-BaoNguyen-7047059.mp3"
    
    # Test Case 1: Default thresholds
    print("\n=== Test Case 1: Default Thresholds ===")
    are_similar = compare_songs(song1_path, song2_path)
    print(f"Are {os.path.basename(song1_path)} and {os.path.basename(song2_path)} similar? {are_similar}")

    # Test Case 2: Custom thresholds (more lenient)
    print("\n=== Test Case 2: Custom Thresholds (Song1 vs Song3) ===")
    are_similar_custom = compare_songs(
        song1_path,
        song3_path,
        mfcc_threshold=0.90,    # Less strict timbre similarity
        tempo_threshold=15,     # Allow 15 BPM difference
        centroid_threshold=350  # Allow 350 Hz difference
    )
    print(f"Are {os.path.basename(song1_path)} and {os.path.basename(song3_path)} similar (custom thresholds)? {are_similar_custom}")

    # Test Case 3: Custom thresholds (Song2 vs Song3)
    print("\n=== Test Case 3: Custom Thresholds (Song2 vs Song3) ===")
    are_similar_custom = compare_songs(
        song2_path,
        song3_path,
        mfcc_threshold=0.90,    # Less strict timbre similarity
        tempo_threshold=15,     # Allow 15 BPM difference
        centroid_threshold=350  # Allow 350 Hz difference
    )
    print(f"Are {os.path.basename(song2_path)} and {os.path.basename(song3_path)} similar (custom thresholds)? {are_similar_custom}")
