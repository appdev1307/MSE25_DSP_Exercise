import librosa
import librosa.display
import numpy as np
import os
from pathlib import Path
from IPython.display import display, HTML
import matplotlib.pyplot as plt

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
    """Extract audio features for comparison and return intermediate results for visualization."""
    try:
        # MFCCs (timbre)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
                
        # Tempo (rhythm)
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
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
        return features, mfcc, beat_frames, spectral_centroid
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None, None, None, None

def compare_songs(input_file, compare_file, mfcc_threshold=0.95, tempo_threshold=5, centroid_threshold=200):
    """
    Compare two audio files based on MFCC, tempo, and spectral centroid.
    Returns tuple (is_similar, mfcc_similarity, tempo_diff, centroid_diff).
    Also visualizes the spectra of the features.
    """
    print(f"\nComparing {input_file} and {compare_file}...")

    # Load audio files
    audio1, sr1 = load_audio(input_file)
    audio2, sr2 = load_audio(compare_file)
    
    if audio1 is None or audio2 is None:
        print("Comparison failed due to loading errors.")
        return False, 0.0, 0.0, 0.0
    
    # Extract features and intermediate results for visualization
    features1, mfcc1, beat_frames1, spectral_centroid1 = extract_features(audio1, sr1)
    features2, mfcc2, beat_frames2, spectral_centroid2 = extract_features(audio2, sr2)
    
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
        
        # Determine similarity with colored HTML output
        is_similar = (
            mfcc_similarity >= mfcc_threshold and
            tempo_diff <= tempo_threshold and
            centroid_diff <= centroid_threshold
        )
        
        result_text = "Similar" if is_similar else "Different"
        color = "green" if is_similar else "red"
        display(HTML(f"Result: <span style='color:{color}'>{result_text}</span>"))

        # --- Visualization ---

        # Convert beat frames to time (in seconds)
        beat_times1 = librosa.frames_to_time(beat_frames1, sr=sr1)
        beat_times2 = librosa.frames_to_time(beat_frames2, sr=sr2)

        # Compute onset strength envelopes for tempo visualization
        onset_env1 = librosa.onset.onset_strength(y=audio1, sr=sr1)
        onset_env2 = librosa.onset.onset_strength(y=audio2, sr=sr2)
        times1 = librosa.times_like(onset_env1, sr=sr1)
        times2 = librosa.times_like(onset_env2, sr=sr2)

        # Time axis for spectral centroids
        centroid_times1 = librosa.times_like(spectral_centroid1, sr=sr1)
        centroid_times2 = librosa.times_like(spectral_centroid2, sr=sr2)

        # Create a figure with subplots
        plt.figure(figsize=(12, 12))

        # 1. MFCCs for both audio files
        plt.subplot(3, 2, 1)
        librosa.display.specshow(mfcc1, x_axis='time', sr=sr1, cmap='viridis')
        plt.colorbar(label='Amplitude (dB)')
        plt.title(f'MFCCs - {Path(input_file).name}')
        plt.xlabel('Time (s)')
        plt.ylabel('MFCC Coefficient Index')

        plt.subplot(3, 2, 2)
        librosa.display.specshow(mfcc2, x_axis='time', sr=sr2, cmap='viridis')
        plt.colorbar(label='Amplitude (dB)')
        plt.title(f'MFCCs - {Path(compare_file).name}')
        plt.xlabel('Time (s)')
        plt.ylabel('MFCC Coefficient Index')

        # 2. Onset Strength Envelopes with Beats (for Tempo)
        plt.subplot(3, 2, 3)
        plt.plot(times1, onset_env1, label='Onset Strength', color='b')
        plt.vlines(beat_times1, 0, onset_env1.max(), color='r', linestyle='--', label='Detected Beats')
        plt.title(f'Onset Strength - {Path(input_file).name} (Tempo: {features1["tempo"]:.2f} BPM)')
        plt.xlabel('Time (s)')
        plt.ylabel('Onset Strength')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 2, 4)
        plt.plot(times2, onset_env2, label='Onset Strength', color='b')
        plt.vlines(beat_times2, 0, onset_env2.max(), color='r', linestyle='--', label='Detected Beats')
        plt.title(f'Onset Strength - {Path(compare_file).name} (Tempo: {features2["tempo"]:.2f} BPM)')
        plt.xlabel('Time (s)')
        plt.ylabel('Onset Strength')
        plt.legend()
        plt.grid(True)

        # 3. Spectral Centroids
        plt.subplot(3, 2, 5)
        plt.plot(centroid_times1, spectral_centroid1, label='Spectral Centroid', color='g')
        plt.axhline(features1['spectral_centroid'], color='r', linestyle='--', label=f'Mean: {features1["spectral_centroid"]:.2f} Hz')
        plt.title(f'Spectral Centroid - {Path(input_file).name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 2, 6)
        plt.plot(centroid_times2, spectral_centroid2, label='Spectral Centroid', color='g')
        plt.axhline(features2['spectral_centroid'], color='r', linestyle='--', label=f'Mean: {features2["spectral_centroid"]:.2f} Hz')
        plt.title(f'Spectral Centroid - {Path(compare_file).name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.legend()
        plt.grid(True)

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()
        
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
    
    # Display summary with colored HTML output
    print("\n=== Comparison Summary ===")
    summary_html = "<table>"
    summary_html += "<tr><th>File</th><th>Result</th><th>MFCC</th><th>Tempo Diff</th><th>Centroid Diff</th></tr>"
    for file_name, is_similar, mfcc_sim, tempo_diff, centroid_diff in results:
        result_text = "Similar" if is_similar else "Different"
        color = "green" if is_similar else "red"
        summary_html += (
            f"<tr>"
            f"<td>{file_name}</td>"
            f"<td><span style='color:{color}'>{result_text}</span></td>"
            f"<td>{mfcc_sim:.3f}</td>"
            f"<td>{tempo_diff:.2f}</td>"
            f"<td>{centroid_diff:.2f}</td>"
            f"</tr>"
        )
    summary_html += "</table>"
    display(HTML(summary_html))

# Example usage
if __name__ == "__main__":
    # Input file and folder path
    input_file = "./MSE25_DSP_data/mp3_store/ConDuongXuaEmDi-HoangThucLinh-5217163.mp3"
    folder_path = "./MSE25_DSP_data/mp3_store"

    # Test Case: Custom thresholds (more lenient)
    print("\n=== Test Case: Custom Thresholds (More Lenient) ===")
    find_and_compare_mp3(
        input_file,
        folder_path,
        mfcc_threshold=0.90,    # Less strict timbre similarity
        tempo_threshold=15,     # Allow 15 BPM difference
        centroid_threshold=350  # Allow 350 Hz difference
    )
