import os
import json
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    # Đường dẫn
    mp3_root = Path("fma_small")
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
                title = mp3_file.name.split('.')[0]

            try:
                artist = df.loc[track_id, ('artist', 'name')]
            except Exception as e:
                artist = "Unknown Artist"

            try:
                album = df.loc[track_id, ('album', 'title')]
            except Exception as e:
                album = "Unknown Album"

            print(f"🔍 Đang xử lý : {mp3_file.name}, trích xuất mfcc, tempo, centroid...")
            audio, sr = librosa.load(mp3_file, sr=None)

            # Tính MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_mean = mfcc.mean(axis=1).tolist()

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