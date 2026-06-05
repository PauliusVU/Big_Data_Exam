import requests
import os

def download_ais_december_2021(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "aisdk-2021-12.zip")
    url = "http://aisdata.ais.dk/2021/aisdk-2021-12.zip"

    if os.path.exists(zip_path):
        print("Zip already exists, skipping download.")
        return

    print(f"Downloading AIS data from {url}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                gb = downloaded / 1024/1024/1024
                total_gb = total / 1024/1024/1024
                print(f"  Progress: {pct:.1f}% ({gb:.2f} GB / {total_gb:.2f} GB)", end="\r")

    print(f"\nDownload complete. Saved to {zip_path}")

if __name__ == "__main__":
    download_ais_december_2021()