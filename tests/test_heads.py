"""M10 heads tests — harness validation ONLY (regime S): the scaffolds train, separate the
synthetic classes they were built to separate, and enforce their split discipline. No claim
about real-world accuracy is made or implied by these tests (Stage B fills the heads)."""
from __future__ import annotations

import numpy as np
import pytest

from bball.heads.audio import (
    AudioHead,
    extract_rim_window,
    log_mel,
    synth_impulse_clip,
)
from bball.heads.fewshot import (
    PrototypicalHead,
    dtw_distance,
    episodic_eval,
    ncm_dtw_predict,
    stats_embed,
)
from bball.heads.shot_type import (
    TwoFeatureLogistic,
    build_temporal_cnn,
    dribble_features,
    feature_vector,
    predict_temporal_cnn,
    synth_prerelease_sequences,
    train_temporal_cnn,
)


# --------------------------------------------------------------------------- #
# T4 shot type
# --------------------------------------------------------------------------- #
def test_dribble_features_separate_classes():
    X, y = synth_prerelease_sequences(20, seed=0)
    band = np.array([dribble_features(x, 60.0)["band_energy"] for x in X])
    # dribble sequences carry far more 1-3 Hz band energy than holds
    assert band[y == 1].mean() > 2 * band[y == 0].mean()


def test_two_feature_logistic_high_accuracy_on_synth():
    Xs, y = synth_prerelease_sequences(40, seed=1)
    F = np.stack([feature_vector(x, 60.0) for x in Xs])
    ntr = 60
    clf = TwoFeatureLogistic().fit(F[:ntr], y[:ntr])
    acc = float((clf.predict(F[ntr:]) == y[ntr:]).mean())
    assert acc >= 0.9        # the plan's "a feature, not a network" claim, on synth (S)


def test_temporal_cnn_learns_synth():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    Xs, y = synth_prerelease_sequences(24, seed=2)
    X = Xs[:, None, :]       # (N, C=1, T)
    ntr = 32
    model = build_temporal_cnn(in_channels=1, base_ch=8)
    losses = train_temporal_cnn(model, X[:ntr], y[:ntr], epochs=40, lr=1e-2, seed=0)
    assert losses[-1] < losses[0]
    acc = float((predict_temporal_cnn(model, X[ntr:]) == y[ntr:]).mean())
    assert acc >= 0.8


def test_dribble_features_handles_short_input():
    f = dribble_features(np.array([1.8, 1.8]), 60.0)
    assert f["band_energy"] == 0.0


# --------------------------------------------------------------------------- #
# Few-shot
# --------------------------------------------------------------------------- #
def _fewshot_items(n_groups=4, n_per=6, seed=0):
    """Two synthetic classes with a group (session) structure: class A = low-freq sine,
    class B = high-freq sine, group-specific amplitude (a venue-like nuisance)."""
    rng = np.random.default_rng(seed)
    items = []
    t = np.linspace(0, 2, 80)
    for g in range(n_groups):
        amp = rng.uniform(0.8, 1.2)
        for _ in range(n_per):
            items.append({"seq": amp * np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, 0.05, len(t)),
                          "label": "A", "group": f"g{g}"})
            items.append({"seq": amp * np.sin(2 * np.pi * 4.0 * t) + rng.normal(0, 0.05, len(t)),
                          "label": "B", "group": f"g{g}"})
    return items


def test_prototypical_head_predicts_and_adds_class():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.1, (10, 4)) + np.array([1, 0, 0, 0])
    b = rng.normal(0, 0.1, (10, 4)) + np.array([0, 1, 0, 0])
    head = PrototypicalHead().fit(np.vstack([a, b]), ["a"] * 10 + ["b"] * 10)
    assert head.predict(np.array([[1, 0, 0, 0.05]])) == ["a"]
    head.add_class("c", rng.normal(0, 0.1, (5, 4)) + np.array([0, 0, 1, 0]))
    assert head.predict(np.array([[0, 0, 1, 0]])) == ["c"]  # no-retraining class addition


def test_dtw_zero_for_identical_and_orders_similarity():
    x = np.sin(np.linspace(0, 3, 50))
    assert dtw_distance(x, x) == pytest.approx(0.0)
    shifted = np.sin(np.linspace(0.15, 3.15, 50))
    other = np.cos(np.linspace(0, 9, 50))
    assert dtw_distance(x, shifted) < dtw_distance(x, other)


def test_episodic_eval_proto_beats_chance_and_respects_groups():
    items = _fewshot_items()
    res = episodic_eval(items, k_shot=3, n_query=3, n_episodes=10, method="proto",
                        embed_fn=stats_embed, seed=0)
    assert res["n_episodes"] > 0
    assert res["accuracy"] > 0.8            # separable classes; harness works (S)
    with pytest.raises(ValueError):          # single group must be refused (leakage guard)
        episodic_eval([dict(it, group="only") for it in items], k_shot=3, n_query=3,
                      method="proto", embed_fn=stats_embed)


def test_ncm_dtw_baseline_runs():
    items = _fewshot_items(n_groups=2, n_per=3)
    res = episodic_eval(items, k_shot=2, n_query=2, n_episodes=4, method="ncm_dtw", seed=1)
    assert res["n_episodes"] > 0
    assert res["accuracy"] > 0.7


# --------------------------------------------------------------------------- #
# T6 audio scaffold
# --------------------------------------------------------------------------- #
def test_rim_window_centers_impulse():
    sr = 16000
    clip = synth_impulse_clip(sr, dur_s=2.0, impulse_t=1.2, kind="click", seed=0)
    win = extract_rim_window(clip, sr, rim_arrival_s=1.2, pre_s=0.15, post_s=0.35)
    assert len(win) == int(0.5 * sr)
    # the impulse (max |amplitude|) sits near the pre_s mark inside the window
    peak = np.argmax(np.abs(win)) / sr
    assert abs(peak - 0.15) < 0.05


def test_log_mel_localizes_impulse_energy():
    sr = 16000
    clip = synth_impulse_clip(sr, dur_s=1.0, impulse_t=0.6, kind="ring", seed=1)
    lm = log_mel(clip, sr, n_fft=512, hop=128, n_mels=32)
    assert lm.shape[0] == 32
    frame_peak = int(np.argmax(lm.max(axis=0)))
    t_peak = frame_peak * 128 / sr
    assert abs(t_peak - 0.6) < 0.08          # energy lands at the impulse time


def test_audio_head_plumbing_separates_click_vs_ring():
    """Plumbing test ONLY: click vs ring are stand-ins proving the window->mel->embed->head
    path trains. This is NOT a swish/rattle accuracy claim (T6 has none in Stage A)."""
    sr = 16000
    clips, labels = [], []
    for i in range(16):
        clips.append(synth_impulse_clip(sr, 0.6, 0.3, kind="click", seed=i)); labels.append(0)
        clips.append(synth_impulse_clip(sr, 0.6, 0.3, kind="ring", seed=100 + i)); labels.append(1)
    head = AudioHead().fit(clips[:24], np.array(labels[:24]), sr)
    p = head.predict_proba(clips[24:], sr)
    acc = float(((p >= 0.5).astype(int) == np.array(labels[24:])).mean())
    assert acc >= 0.85
