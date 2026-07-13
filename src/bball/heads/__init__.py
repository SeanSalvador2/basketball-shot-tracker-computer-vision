"""HEADS — small learned layers on top of geometry (harness-validated in Stage A).

shot_type.py  dribble-oscillation feature + 2-feature logistic + small 1D-CNN head
fewshot.py    prototypical head over embeddings + episodic eval; NCM-DTW baseline
audio.py      log-mel windowing + embedding-head interface (scaffold; no accuracy claims)

All heads are trainable on synthetic sequences as harness validation only; Stage B fills
them with real data. Regime labels (S/R) accompany every reported number.
"""
from bball.heads.audio import AudioHead, extract_rim_window, log_mel, synth_impulse_clip
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

__all__ = [
    "dribble_features", "feature_vector", "TwoFeatureLogistic",
    "build_temporal_cnn", "train_temporal_cnn", "predict_temporal_cnn",
    "synth_prerelease_sequences",
    "PrototypicalHead", "dtw_distance", "ncm_dtw_predict", "episodic_eval", "stats_embed",
    "AudioHead", "log_mel", "extract_rim_window", "synth_impulse_clip",
]
