import os
import json
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

def extract_mfcc_pro(file_path,
                     sr=44100,
                     n_mfcc=40,
                     n_fft=4096,
                     hop_length=256,
                     use_delta=True):
    """
    Trích xuất MFCC chuyên sâu:
    - Chuẩn hóa âm lượng
    - Cắt khoảng lặng
    - Lọc pre-emphasis
    - MFCC + delta (biến động)
    """

    # Load audio với sampling rate cao
    y, sr = librosa.load(file_path, sr=sr)

    # Normalize biên độ
    y = librosa.util.normalize(y)

    # Cắt khoảng lặng (silence)
    y, _ = librosa.effects.trim(y)

    # Pre-emphasis filter (làm nổi bật tần số cao)
    pre_emphasis = 0.97
    y_preemph = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # Tính MFCC
    mfcc = librosa.feature.mfcc(
        y=y_preemph,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )

    if use_delta:
        # Tính delta và delta-delta
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_all = np.vstack([mfcc, delta, delta2])
    else:
        mfcc_all = mfcc

    # Trả về vector trung bình
    mfcc_mean = mfcc_all.mean(axis=1)

    return mfcc_mean

# Tính MFCC siêu cấp (ultra) với độ chính xác tối đa
def extract_mfcc_ultra(file_path,
                       sr=48000,
                       n_mfcc=40,
                       n_fft=4096,
                       hop_length=256,
                       use_delta=True,
                       apply_cmvn=False):
    """
    Trích xuất MFCC siêu cấp: độ chính xác tối đa
    """

    # Load & normalize
    y, _ = librosa.load(file_path, sr=sr)
    y = librosa.util.normalize(y)

    # Trim silence (chính xác hơn)
    y, _ = librosa.effects.trim(y, top_db=30)

    # Pre-emphasis filter
    pre_emphasis = 0.97
    y_preemph = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y_preemph,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # Thêm log-energy (dòng đầu tiên là tổng năng lượng khung)
    log_energy = librosa.feature.rms(y=y_preemph, hop_length=hop_length).flatten()
    log_energy = np.log1p(log_energy)  # log(1 + E)
    log_energy = np.expand_dims(log_energy, axis=0)
    mfcc = np.vstack([log_energy, mfcc])  # giờ là (n_mfcc+1, T)

    if use_delta:
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc = np.vstack([mfcc, delta, delta2])  # (n_feat*3, T)

    if apply_cmvn:
        # Apply Cepstral Mean Variance Normalization
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / \
               (np.std(mfcc, axis=1, keepdims=True) + 1e-6)

    mfcc_mean = mfcc.mean(axis=1)
    return mfcc_mean


if __name__ == '__main__':
    # Đường dẫn
    # mp3_root = Path("fma_small")
    mp3_root = Path("custom_database")
    metadata_path = Path("fma_metadata/tracks.csv")
    output_file = "fma_20_songs_db.json"

    # Đọc metadata
    df = pd.read_csv(metadata_path, header=[0, 1], index_col=0)

    # Lấy 20 bài đầu tiên từ fma_small
    mp3_files = []
    for root, _, files in os.walk(mp3_root):
        for file in files:
            if file.endswith(".mp3"):
                mp3_files.append(Path(root) / file)
    mp3_files = sorted(mp3_files)[:20] # Lấy 20 bài đầu tiên (Remove :20 to get all)
    # mp3_files = sorted(mp3_files) # Lấy tất cả các bài trong thư mục

    # Xử lý từng bài và lưu đặc trưng
    db = []
    for mp3_file in mp3_files:
        try:
            try:
                track_id = int(mp3_file.stem)
            except Exception as e:
                track_id = "-1"
            try:
                title = df.loc[track_id, ('track', 'title')]
            except Exception as e:
                title = mp3_file.name.split('-')[0]

            try:
                artist = df.loc[track_id, ('artist', 'name')]
            except Exception as e:
                artist = mp3_file.name.split('-')[1]

            try:
                album = df.loc[track_id, ('album', 'title')]
            except Exception as e:
                album = "Unknown Album"

            print(f"🔍 Đang xử lý : {mp3_file.name}, trích xuất mfcc, tempo, centroid...")
            audio, sr = librosa.load(mp3_file, sr=None)

            # Tính MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            # mfcc_mean = mfcc.mean(axis=1).tolist() # Normal MFCC
            mfcc_mean = extract_mfcc_pro(mp3_file).tolist() # Pro MFCC
            # mfcc_mean = extract_mfcc_ultra(mp3_file).tolist() # Ultra MFCC

            # Tempo (rhythm)
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            tempo = float(tempo) if isinstance(tempo, np.ndarray) else tempo

            # Spectral Centroid (brightness/tone)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            centroid_mean = np.mean(spectral_centroid)

            db.append({
                "track_id": track_id,
                "title": title,
                "filename": str(mp3_file),
                "artist_name": artist,
                "album_title": album,
                "mfcc": mfcc_mean,
                "tempo": tempo,
                "spectral_centroid": centroid_mean,
            })

        except Exception as e:
            print(f"⚠️ Lỗi xử lý {mp3_file.name}: {e}")

    # Ghi ra file JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Đã lưu vào: {output_file}")