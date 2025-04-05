import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

# === GHI ÂM ===
def record_audio(filename="recorded.wav", duration=10, fs=44100):
    print("🎙️ Đang ghi âm...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print(f"✅ Ghi xong. Lưu tại {filename}")

# === TRÍCH MFCC + TEMPO + CENTROID ===
def extract_features(file_path):
    """Extract MFCC, tempo, and spectral centroid from an audio file."""
    print("extract_features of", file_path)
    audio, sr = librosa.load(file_path, sr=None)

    # Tính MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    # mfcc_mean = mfcc.mean(axis=1).toList() # Normal MFCC
    mfcc_mean = extract_mfcc_pro(file_path).tolist() # Pro MFCC
    # mfcc_mean = extract_mfcc_ultra(file_path).tolist() # Ultra MFCC

    # Tempo (rhythm)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo = float(tempo) if isinstance(tempo, np.ndarray) else tempo

    # Spectral Centroid (brightness/tone)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    centroid_mean = np.mean(spectral_centroid)

    return mfcc_mean, tempo, centroid_mean


# === TẢI DATABASE ===
def load_database(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

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




# === SO KHỚP MFCC + TEMPO + CENTROID ===
# 🎯 Nhận diện cực chính xác (bản gốc 100%)	MFCC ≥ 0.95, tempo ≤ 5, centroid ≤ 150
# 🎵 Nhận diện bản phối lại	                MFCC ≥ 0.85, tempo ≤ 20, centroid ≤ 500
# 🎙️ Nhận diện người huýt sáo/hát lại	    MFCC ≥ 0.75, tempo ≤ 25, centroid bỏ qua
def find_best_match_whistle(query_vec, query_tempo, query_centroid, database,
                            mfcc_weight=0.65, tempo_weight=0.25, centroid_weight=0.1):
    candidates = []

    for song in database:
        db_vec = np.array(song["mfcc"])


        sim_mfcc = np.dot(query_vec, song['mfcc']) / (
                np.linalg.norm(query_vec) * np.linalg.norm(song['mfcc'])
        )

        # sim_mfcc = np.dot(query_vec, db_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(db_vec))
        # sim_mfcc = cosine_similarity([query_vec], [db_vec])[0][0]

        tempo_diff = abs(query_tempo - song["tempo"])
        centroid_diff = abs(query_centroid - song["spectral_centroid"])

        # Debug output
        print(f"\n🎵 {song['title']} - {song['artist_name']}")
        print(f"MFCC: {sim_mfcc:.3f}, Tempo Diff: {tempo_diff:.2f} BPM, Centroid Diff: {centroid_diff:.2f} Hz")

        # Sử dụng threshold gần sát 100%
        # if sim_mfcc >= 0.98 or tempo_diff <= 2 or centroid_diff <= 50:
        #     candidates.append((song, 0.98)
        #     continue

        # Lọc theo threshold
        if sim_mfcc >= 0.75 and tempo_diff <= 25 and centroid_diff <= 500:
            # Chuẩn hoá các diff về [0–1] rồi chuyển thành điểm
            tempo_score = 1 - (tempo_diff / 25)
            centroid_score = 1 - (centroid_diff / 500)

            score = sim_mfcc * mfcc_weight + tempo_score * tempo_weight + centroid_score * centroid_weight
            candidates.append((song, score))

    # Trả về bài có tổng điểm cao nhất
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        print("\n🔍 Các bài hát khả thi:")
        for song, score in candidates:
            print(f"  - {song['title']} - {song['artist_name']}: {score:.3f}")

        best_song = candidates[0][0]
        print(f"\n✅ Nhận diện: {best_song['title']} - {best_song['artist_name']}")
        return best_song
    else:
        print("\n❌ Không có bài nào đạt đủ điều kiện threshold.")
        return None


# === CHẠY TOÀN BỘ ===
if __name__ == "__main__":
    record_audio()
    mfcc, tempo, centroid = extract_features("recorded.wav")
    db = load_database("fma_20_songs_db.json")
    find_best_match_whistle(mfcc, tempo, centroid, db)

