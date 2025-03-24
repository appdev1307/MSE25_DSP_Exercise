import librosa
import numpy as np
import os
import sys
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

def compare_songs(input_file, compare_file, mfcc_threshold=0.95, tempo_threshold=5, centroid_threshold=200):
    """
    Compare two audio files based on MFCC, tempo, and spectral centroid.
    Returns tuple (is_similar, mfcc_similarity, tempo_diff, centroid_diff).
    """
    print(f"\nComparing {input_file} and {compare_file}...")

    # Load audio files
    audio1, sr1 = load_audio(input_file)
    audio2, sr2 = load_audio(compare_file)
    
    if audio1 is None or audio2 is None:
        print("Comparison failed due to loading errors.")
        return False, 0.0, 0.0, 0.0
    
    # Extract features
    features1 = extract_features(audio1, sr1)
    features2 = extract_features(audio2, sr2)
    
    if features1 is None or features2 is None:
        print("Comparison failed due to feature extraction errors.")
        return False, 0.0, 0.0, 0.0
    
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
        return is_similar, mfcc_similarity, tempo_diff, centroid_diff
    except Exception as e:
        print(f"Error during comparison: {e}")
        return False, 0.0, 0.0, 0.0

def find_and_compare_mp3(input_file, folder_path, **kwargs):
    """Find all .mp3 files in a folder and compare them with the input file."""
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} does not exist.")
        return

    if not Path(folder_path).is_dir():
        print(f"Error: Folder {folder_path} does not exist or is not a directory.")
        return

    # Find all .mp3 files in the folder
    mp3_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp3')]
    if not mp3_files:
        print(f"No .mp3 files found in {folder_path}.")
        return

    print(f"\nFound {len(mp3_files)} .mp3 files in {folder_path}:")
    for mp3_file in mp3_files:
        print(f" - {mp3_file}")

    # Compare input file with each .mp3 file in the folder
    results = []
    for mp3_file in mp3_files:
        compare_file_path = os.path.join(folder_path, mp3_file)
        is_similar, mfcc_sim, tempo_diff, centroid_diff = compare_songs(input_file, compare_file_path, **kwargs)
        results.append((mp3_file, is_similar, mfcc_sim, tempo_diff, centroid_diff))
    
    # Display summary
    print("\n=== Comparison Summary ===")
    for file_name, is_similar, mfcc_sim, tempo_diff, centroid_diff in results:
        print(f"{file_name}: {'Similar' if is_similar else 'Different'} "
              f"(MFCC: {mfcc_sim:.3f}, Tempo Diff: {tempo_diff:.2f}, Centroid Diff: {centroid_diff:.2f})")

# Example usage with command-line argument
if __name__ == "__main__":
    # Check if input file is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file_path>")
        sys.exit(1)

    # Get input file from command-line argument
    input_file = sys.argv[1]
    folder_path = "./MSE25_DSP_data"  # Folder path can still be hardcoded or also passed as an argument

    # Test Case 1: Default thresholds
    print("\n=== Test Case 1: Default Thresholds ===")
    find_and_compare_mp3(
        input_file,
        folder_path,
        mfcc_threshold=0.95,
        tempo_threshold=5,
        centroid_threshold=200
    )

    # Test Case 2: Custom thresholds (more lenient)
    print("\n=== Test Case 2: Custom Thresholds (More Lenient) ===")
    find_and_compare_mp3(
        input_file,
        folder_path,
        mfcc_threshold=0.90,    # Less strict timbre similarity
        tempo_threshold=15,     # Allow 15 BPM difference
        centroid_threshold=350  # Allow 350 Hz difference
    )
