import os
import kagglehub
from pathlib import Path
import zipfile


def ensure_kaggle_auth():
    """
    check for Kaggle credentials in the following order:
    1. Environment variables KAGGLE_USERNAME and KAGGLE_KEY
    2. ~/.kaggle/kaggle.json
    if not found, prompt the user to login interactively using kagglehub.login()
    """

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if username and key:
        print("Using Kaggle credentials from environment variables.")
        return

    # check ~/.kaggle/kaggle.json
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        print("Using Kaggle credentials from ~/.kaggle/kaggle.json")
        return

    # fallback：interactive login
    print("Kaggle credentials not found. Starting login...")
    kagglehub.login()


def download():
    ensure_kaggle_auth()

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    raw_dir = BASE_DIR / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    path = kagglehub.dataset_download("blastchar/telco-customer-churn", output_dir=raw_dir)
    
    if Path(path).suffix == ".zip":
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        path.unlink()  # remove the zip file after extraction

    print(f"Dataset downloaded to {raw_dir.resolve()}")


if __name__ == "__main__":
    download()