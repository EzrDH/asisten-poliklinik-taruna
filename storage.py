import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CONFIG = {"baseline_hari": 14, "z_ambang": 2.0, "c_min": 3}


def _path(name):
    return Path(DATA_DIR) / name


def _load(name, default):
    p = _path(name)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(name, obj):
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    with open(_path(name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_master():
    return _load("master_taruna.json", [])


def save_master(data):
    _save("master_taruna.json", data)


def load_kunjungan():
    return _load("kunjungan.json", [])


def save_kunjungan(data):
    _save("kunjungan.json", data)


def load_config():
    cfg = _load("config.json", None)
    return cfg if cfg else dict(DEFAULT_CONFIG)


def tambah_kunjungan(record):
    data = load_kunjungan()
    record["id"] = max([r.get("id", 0) for r in data], default=0) + 1
    data.append(record)
    save_kunjungan(data)
    return record
