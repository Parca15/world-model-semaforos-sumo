"""Pruebas de los modelos, la evaluación predictiva y la estadística."""
import numpy as np
import torch

from wm.eval.rollout import rollout
from wm.eval.stats import bootstrap_ci, cliffs_delta, holm, paired_comparisons
from wm.models.baselines import MovingAverage, Persistence, Ridge
from wm.models.predictor import TorchPredictor
from wm.models.tsmixer import build_tsmixer, count_params

SD, NT, W = 91, 7, 12


def test_tsmixer_shapes_and_budget():
    for nb, d, norm in [(2, 64, "layer"), (4, 96, "batch"), (6, 128, "layer")]:
        m = build_tsmixer(150_000, n_blocks=nb, d_model=d, norm=norm)
        assert abs(count_params(m) - 150_000) / 150_000 < 0.10
        m.eval()
        delta, r = m(torch.randn(3, W, SD + NT))
        assert delta.shape == (3, SD) and r.shape == (3, NT)


def test_tsmixer_uses_the_whole_window():
    """Cambiar el primer paso de la ventana debe cambiar la predicción (hay mezcla temporal)."""
    torch.manual_seed(0)
    m = build_tsmixer(150_000).eval()
    x = torch.randn(1, W, SD + NT)
    x2 = x.clone()
    x2[:, 0] += 1.0
    assert not torch.allclose(m(x)[0], m(x2)[0])


def test_persistence_rollout_keeps_state():
    x0 = np.random.default_rng(0).normal(size=(5, W, SD + NT)).astype(np.float32)
    states, rewards = rollout(Persistence(SD, NT), x0, np.zeros((5, 3, NT), np.float32), 4)
    assert np.allclose(states, x0[:, -1:, :SD])
    assert np.allclose(rewards, 0)


def test_rollout_feeds_predictions_back():
    """Con Δ constante c, el estado a h pasos debe ser s_t + h·c."""
    class ConstDelta:
        name, state_dim = "c", SD

        def predict(self, x):
            return np.full((len(x), SD), 0.5, np.float32), np.zeros((len(x), NT), np.float32)

    x0 = np.zeros((2, W, SD + NT), np.float32)
    states, _ = rollout(ConstDelta(), x0, np.zeros((2, 4, NT), np.float32), 5)
    assert np.allclose(states[:, -1], 2.5)


def test_moving_average_and_ridge():
    x = np.random.default_rng(1).normal(size=(200, W, SD + NT)).astype(np.float32)
    d, _ = MovingAverage(SD, NT).predict(x)
    assert np.allclose(d, x[:, :, :SD].mean(1) - x[:, -1, :SD], atol=1e-5)
    # ridge con alpha ~0 recupera un mapa lineal exacto
    true = np.random.default_rng(2).normal(size=(W * (SD + NT) + 1, SD + NT)).astype(np.float32) * 0.01
    X = Ridge._design(x)
    Y = X @ true
    ridge = Ridge(SD, alpha=1e-8).fit_gram(X.T @ X, X.T @ Y)
    pd_, pr = ridge.predict(x)
    assert np.allclose(np.concatenate([pd_, pr], 1), Y, atol=1e-3)


def test_torch_predictor_batches_consistently():
    m = build_tsmixer(150_000, norm="layer").eval()
    x = np.random.default_rng(3).normal(size=(10, W, SD + NT)).astype(np.float32)
    a = TorchPredictor(m, "t", SD, batch_size=3).predict(x)
    b = TorchPredictor(m, "t", SD, batch_size=100).predict(x)
    assert np.allclose(a[0], b[0], atol=1e-5)


def test_stats_helpers():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert cliffs_delta([3, 4, 5], [0, 1, 2]) == 1.0
    assert cliffs_delta([1, 2], [1, 2]) == 0.0
    mean, lo, hi = bootstrap_ci(np.arange(100), n=2000)
    assert lo < mean < hi
    rows = paired_comparisons("A", {"A": np.arange(10) * 1.0, "B": np.arange(10) + 5.0})
    assert rows[0]["referencia_mejor"] and rows[0]["p_holm"] < 0.05


def test_multistep_loss_uses_future_actions_and_is_finite():
    from wm.data.base import FlatWindows
    from wm.train.trainer import TrainConfig, _loss

    rng = np.random.default_rng(0)
    rows = rng.normal(size=(60, SD + NT)).astype(np.float32)
    delta = rng.normal(size=(60, SD)).astype(np.float32)
    reward = rng.normal(size=(60, NT)).astype(np.float32)
    fw = FlatWindows(rows, delta, reward, np.arange(W - 1, 50), W)
    torch.manual_seed(0)
    m = build_tsmixer(150_000).eval()
    idx = np.arange(8)
    one = _loss(m, fw, idx, TrainConfig(rollout_k=1))[0]
    # con k = 5 el primer término de estado es el mismo error que a 1 paso
    x, d, r, fut = fw.batch(idx, 5)
    assert d.shape == (8, 5, SD) and fut.shape == (8, 5, SD + NT)
    assert torch.allclose(fut[:, 0], torch.from_numpy(rows[fw.ends[idx]]))
    five = _loss(m, fw, idx, TrainConfig(rollout_k=5))[0]
    assert torch.isfinite(five) and not torch.isclose(one, five)


def test_interrupted_training_resumes_to_the_same_result(tmp_path):
    """Entrenar 3 épocas, "interrumpir" y reanudar hasta 6 debe dar lo mismo que 6 épocas seguidas."""
    from wm.data.base import FlatWindows
    from wm.train.trainer import TrainConfig, fit

    rng = np.random.default_rng(0)
    rows = rng.normal(size=(80, SD + NT)).astype(np.float32)
    fw = FlatWindows(rows, rng.normal(size=(80, SD)).astype(np.float32),
                     rng.normal(size=(80, NT)).astype(np.float32), np.arange(W - 1, 70), W)

    def train(epochs, state_path=None, tag="a"):
        torch.manual_seed(0)
        m = build_tsmixer(150_000, n_blocks=2, d_model=64, dropout=0.1)
        cfg = TrainConfig(max_epochs=epochs, batch_size=16, patience=100, samples_per_epoch=32)
        return fit(m, fw, fw, cfg, state_path=state_path, state_tag=tag)

    straight, res = train(6)
    state = tmp_path / "state.pt"
    train(3, state)
    resumed, res2 = train(6, state)
    assert res2.epochs_run == 6 and len(res2.history) == 6
    assert np.isclose(res.best_val_loss, res2.best_val_loss)
    for a, b in zip(straight.state_dict().values(), resumed.state_dict().values()):
        assert torch.allclose(a, b)
    # un estado de otro trabajo (tag distinto) se ignora
    _, res3 = train(2, state, tag="otro")
    assert res3.epochs_run == 2


def test_lstm_and_transformer_shapes_and_budget():
    from wm.models.temporal import build_temporal

    for kind, arch in [("lstm", {"n_layers": 2, "dropout": 0.1}),
                       ("transformer", {"n_layers": 2, "d_model": 64, "n_heads": 4}),
                       ("transformer", {"n_layers": 4, "d_model": 32, "n_heads": 2})]:
        m = build_temporal(kind, 150_000, **arch)
        assert abs(count_params(m) - 150_000) / 150_000 < 0.10, (kind, arch, count_params(m))
        m.eval()
        delta, r = m(torch.randn(3, W, SD + NT))
        assert delta.shape == (3, SD) and r.shape == (3, NT)


def test_lstm_param_formula_matches_torch():
    from wm.models.lstm import LSTMConfig, LSTMModel, lstm_params

    for layers, hidden in [(1, 40), (2, 91), (3, 17)]:
        cfg = LSTMConfig(n_layers=layers, hidden=hidden)
        assert lstm_params(cfg) == count_params(LSTMModel(cfg))


def test_lstm_and_transformer_use_the_whole_window():
    from wm.models.temporal import build_temporal

    torch.manual_seed(0)
    for kind in ("lstm", "transformer"):
        m = build_temporal(kind, 150_000).eval()
        x = torch.randn(1, W, SD + NT)
        x2 = x.clone()
        x2[:, 0] += 1.0
        assert not torch.allclose(m(x)[0], m(x2)[0]), kind


def test_temporal_checkpoints_roundtrip(tmp_path):
    from wm.experiments.common import load_checkpoint, save_checkpoint
    from wm.models.temporal import TEMPORAL_MODELS, build_temporal

    x = torch.randn(2, W, SD + NT)
    for kind in TEMPORAL_MODELS:
        m = build_temporal(kind, 150_000).eval()
        save_checkpoint(m, kind, m.cfg.to_dict(), tmp_path / f"{kind}.pt")
        m2 = load_checkpoint(tmp_path / f"{kind}.pt")
        assert type(m2) is type(m) and torch.allclose(m(x)[0], m2(x)[0])
