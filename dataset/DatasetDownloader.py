import requests
import zipfile
import os
import time

def download_fma_small(retries=3, delay=5):
    url = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
    zip_path = "fma_small.zip"

    if not os.path.exists("fma_small"):
        print("⬇️  Đang tải dataset FMA-small...")
        for attempt in range(retries):
            try:
                r = requests.get(url, stream=True)
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                print("✅ Tải xong. Đang giải nén...")

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall("fma_small")

                print("📂 Giải nén hoàn tất.")
                break
            except (requests.exceptions.RequestException, zipfile.BadZipFile) as e:
                print(f"Error during download: {e}")
                if attempt < retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print("Failed to download after several attempts.")
                    raise
    else:
        print("📦 Dataset đã tồn tại.")

if __name__ == '__main__':
    download_fma_small()