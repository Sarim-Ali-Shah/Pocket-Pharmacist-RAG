# app/data_loader.py
import json
from pathlib import Path
from rank_bm25 import BM25Okapi
from huggingface_hub import hf_hub_download
import shutil
import zipfile
from app.config import TABLES_DIR, HF_REPO, HF_TOKEN

_checked_zips = set()


from app.config import CHUNKS_DIR, CHUNK_VECTORS_DIR, SUBJECTS, HF_TOKEN, HF_REPO

def _ensure_local(remote_name, local_path):
    if local_path.exists():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(repo_id=HF_REPO, filename=remote_name, repo_type="dataset", token=HF_TOKEN)
    shutil.copy(downloaded, local_path)

def load_all_chunks():
    all_chunks = {}
    for subject in SUBJECTS:
        local_path = CHUNKS_DIR / f"{subject}_chunks.json"
        _ensure_local(f"chunks_final/{subject}_chunks.json", local_path)
        with open(local_path, "r", encoding="utf-8") as f:
            all_chunks[subject] = json.load(f)
    return all_chunks

def build_bm25_indexes(all_chunks):
    bm25_indexes = {}
    for subject, chunks in all_chunks.items():
        texts = [c["text"] for c in chunks]
        tokenized = [t.lower().split() for t in texts]
        bm25_indexes[subject] = BM25Okapi(tokenized)
    return bm25_indexes


def ensure_table_image(filename):
    local_path = TABLES_DIR / filename
    if local_path.exists():
        return local_path

    if "_table_" not in filename:
        return None
    prefix = filename.split("_table_")[0]
    zip_remote = f"tables/{prefix}_tables.zip"

    if zip_remote in _checked_zips:
        return local_path if local_path.exists() else None

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(repo_id=HF_REPO, filename=zip_remote, repo_type="dataset", token=HF_TOKEN)
        with zipfile.ZipFile(downloaded, "r") as zf:
            zf.extractall(TABLES_DIR)
    except Exception:
        pass

    _checked_zips.add(zip_remote)
    return local_path if local_path.exists() else None