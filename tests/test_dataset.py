"""Pruebas de la Etapa 3: plan de episodios, construcción, integridad, fugas y cargador común.

Construye un mini-dataset (todos los episodios del plan, pero de 90 s) en un directorio temporal.
"""
import json
import os
import shutil
import stat
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from wm.data.base import load_base
from wm.data.build_base import build, check_no_leakage, plan_episodes
from wm.data.manifest import ManifestError, verify_manifest
from wm.data.normalization import NormStats
from wm.env.state_builder import FEATURES
from wm.utils import load_config

VERSION = "mini"
W, H = 4, 5


@pytest.fixture(scope="session")
def mini(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    root = build(VERSION, workers=4, episode_seconds=90, data_dir=data_dir)
    yield data_dir, root
    for p in root.rglob("*"):  # devolver permisos de escritura para poder borrar el temporal
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)


# ------------------------------------------------------------ plan
def test_plan_matches_design():
    specs = plan_episodes(load_config("dataset"), load_config("base"))
    df = pd.DataFrame([s.__dict__ for s in specs])
    assert len(df) == 104
    assert df.groupby("split").size().to_dict() == {"train": 64, "val": 16, "test": 16, "test_ood": 8}
    assert set(df.loc[df.split != "test_ood", "demand"]) == {"D1", "D2", "D3", "D4"}
    assert set(df.loc[df.split == "test_ood", "demand"]) == {"D5"}
    assert df.episode.is_unique
    p2 = df[df.policy == "P2"]
    assert set(p2.policy_params.map(lambda d: d["p"])) == {0.2, 0.5}


def test_leakage_is_rejected():
    specs = plan_episodes(load_config("dataset"), load_config("base"))
    bad = specs + [replace(specs[0], episode="ep_bad", split="test")]
    with pytest.raises(ValueError):
        check_no_leakage(bad)


# ------------------------------------------------------------ construcción e integridad
def test_files_and_manifest(mini):
    _, root = mini
    for name in ("index.parquet", "splits.json", "feature_schema.json", "norm_stats.json",
                 "config_used.yaml", "MANIFEST.sha256", "README.md"):
        assert (root / name).exists(), name
    assert len(list((root / "episodes").glob("ep_*.npz"))) == 104
    assert verify_manifest(root) == 104 + 6


def test_dataset_is_read_only_and_not_overwritten(mini):
    data_dir, root = mini
    assert not os.access(root / "norm_stats.json", os.W_OK)
    with pytest.raises(FileExistsError):
        build(VERSION, workers=1, episode_seconds=90, data_dir=data_dir)


def test_tampering_is_detected(mini, tmp_path):
    _, root = mini
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    target = copy / "episodes" / "ep_000.npz"
    os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(ManifestError):
        verify_manifest(copy)


def test_no_leakage_between_splits(mini):
    _, root = mini
    index = pd.read_parquet(root / "index.parquet")
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    seeds = {s: set(index.set_index("episode").loc[eps, "seed"]) for s, eps in splits.items()}
    names = list(seeds)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not seeds[a] & seeds[b], (a, b)
    assert sum(len(v) for v in splits.values()) == len(index)


def test_normalization_fitted_only_on_train(mini):
    _, root = mini
    index = pd.read_parquet(root / "index.parquet")
    norm = NormStats.load(root / "norm_stats.json")
    mask = np.array([n for _, _, n in FEATURES])

    def fit(split):
        states, rewards = [], []
        for f in index.loc[index.split.isin(split), "file"]:
            with np.load(root / f) as z:
                states.append(z["states"]), rewards.append(z["rewards"])
        return NormStats.fit(states, rewards, mask)

    assert np.allclose(norm.state_mean, fit(["train"]).state_mean)
    assert not np.allclose(norm.state_mean, fit(["train", "val", "test", "test_ood"]).state_mean)
    assert np.all(norm.state_mean[:, ~mask] == 0) and np.all(norm.state_std[:, ~mask] == 1)


# ------------------------------------------------------------ cargador
def test_windows_never_cross_episodes(mini):
    data_dir, _ = mini
    ds = load_base(VERSION, "val", W=W, H=H, data_dir=data_dir)
    T = ds.episodes[0].T
    assert len(ds) == len(ds.episodes) * (T - W - H + 2)
    for e, t in ds.windows:
        assert t - W + 1 >= 0 and t + H <= ds.episodes[e].T


def test_item_shapes_and_consistency(mini):
    data_dir, _ = mini
    ds = load_base(VERSION, "train", W=W, H=H, data_dir=data_dir)
    item = ds[3]
    assert item["x"].shape == (W, 7 * 13 + 7) and ds.input_dim == 98
    assert item["delta"].shape == (91,) and item["reward"].shape == (7,)
    assert item["future_states"].shape == (H, 91)
    assert item["future_actions"].shape == (H - 1, 7)
    last_state = item["x"][-1, :91]
    assert np.allclose(item["future_states"][0] - last_state, item["delta"], atol=1e-5)
    assert np.allclose(item["future_rewards"][0], item["reward"])
    X, dS, R = load_base(VERSION, "train", W=W, H=1, data_dir=data_dir).arrays()
    assert X.shape[1:] == (W, 98) and dS.shape[1] == 91 and R.shape[1] == 7


def test_train_is_standardized(mini):
    data_dir, _ = mini
    ds = load_base(VERSION, "train", W=1, H=1, data_dir=data_dir)
    s = np.concatenate([ep.states for ep in ds.episodes])
    mask = np.array([n for _, _, n in FEATURES])
    std = s[..., mask].std(0)
    assert np.allclose(s[..., mask].mean(0), 0, atol=1e-3)
    assert np.all((np.abs(std - 1) < 1e-3) | (std == 0))
