"""Locating PanPPI checkpoints: a local file, or a named model downloaded once from the archived record and cached in ~/.cache/panppi."""
import hashlib
import os
import sys
import urllib.request

# Set at release time (Zenodo record file URL prefix) and filled with the md5 of every file.
RECORD_URL = "https://zenodo.org/records/24407809"
WEIGHTS_BASE_URL = os.environ.get("PANPPI_WEIGHTS_URL", RECORD_URL + "/files/")      # works once the record is public
MODELS = ["human_s42", "human_s7", "human_s141", "bernett_s42", "bernett_s7", "bernett_s141"]
ALIASES = {"human": "human_s42", "bernett": "bernett_s42"}
MD5 = {                                                               # md5 of every archived file (verified against the cluster copies)
    "weights_human_s42_best.pt": "95e8d8c032c1873cb33520bffade7d1f",
    "weights_human_s42_calibration.json": "a2cbda9a8edb1a01ebcf546d353fe728",
    "weights_human_s7_best.pt": "3d32fd57d35d7f5b5c9a525a0599a527",
    "weights_human_s7_calibration.json": "4c765e0f82126ccfff7a00ff20ef3400",
    "weights_human_s141_best.pt": "39e0675285e05816cd9f9785a817faae",
    "weights_human_s141_calibration.json": "1d5268b724b3244caed4bfbd2cf4e056",
    "weights_bernett_s42_best.pt": "bc4e76147d4d54c56d0353d2f7bfd6f8",
    "weights_bernett_s42_calibration.json": "07a58f09281ddbd53ce4182807ddd4c1",
    "weights_bernett_s7_best.pt": "c185a03cdd07a3ebfb1403bf7f4f5712",
    "weights_bernett_s7_calibration.json": "550e262e8f0ca55cdd927f786372b56d",
    "weights_bernett_s141_best.pt": "a1ec0ad4f659ae423fd59bb9163f9f03",
    "weights_bernett_s141_calibration.json": "36169d8f9bdc10fd5b2affd6da2b3e03",
}


def cache_dir():
    d = os.environ.get("PANPPI_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "panppi"))
    os.makedirs(d, exist_ok=True)
    return d


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(name, dest):
    url = WEIGHTS_BASE_URL.rstrip("/") + "/" + name + "?download=1"
    print(f"downloading {name} (one time, saved to {dest}) ...", flush=True)
    tmp = dest + ".part"
    def hook(blocks, bs, total):
        if total > 0 and blocks % 400 == 0:
            print(f"  {min(100, int(100 * blocks * bs / total))}%", flush=True)
    try:
        urllib.request.urlretrieve(url, tmp, hook)
    except Exception as e:                                    # unpublished record, no network, ...
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"could not download {name} ({e}). If the archive is not public yet, or you are offline, download it manually from {RECORD_URL} and put it in {os.path.dirname(dest)}, or pass --weights and --calibration.")
    if name in MD5 and _md5(tmp) != MD5[name]:
        os.remove(tmp)
        raise RuntimeError(f"checksum mismatch for {name}; the download was corrupted, please retry")
    os.replace(tmp, dest)


def resolve_weights(model="human", weights=None, calibration=None):
    """Return (checkpoint path, calibration path)."""
    if weights:
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"weights file not found: {weights}")
        cal = calibration or (weights[:-len("_best.pt")] + "_calibration.json" if weights.endswith("_best.pt") else os.path.splitext(weights)[0] + "_calibration.json")
        if not os.path.isfile(cal):
            raise FileNotFoundError(f"calibration file not found: {cal} (pass --calibration)")
        return weights, cal
    name = ALIASES.get(model, model)
    if name not in MODELS:
        raise ValueError(f"unknown model '{model}'. Choose from: human, bernett, {', '.join(MODELS)} (or pass --weights)")
    d = cache_dir()
    files = [f"weights_{name}_best.pt", f"weights_{name}_calibration.json"]
    for f in files:
        p = os.path.join(d, f)
        if not os.path.isfile(p):
            if not WEIGHTS_BASE_URL:
                raise FileNotFoundError(f"{f} is not in {d}. Download it from the PanPPI archive (see the README) and put it there, or pass --weights/--calibration.")
            _download(f, p)
    return os.path.join(d, files[0]), os.path.join(d, files[1])
