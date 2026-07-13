"""HEADS — small learned layers on top of geometry (harness-validated in Stage A).

shot_type.py  dribble-oscillation feature + 2-feature logistic + small 1D-CNN head
fewshot.py    prototypical head over embeddings + episodic eval; NCM-DTW baseline
audio.py      log-mel windowing + embedding-head interface (scaffold; no accuracy claims)

All heads are trainable on synthetic sequences as harness validation only; Stage B fills
them with real data. Regime labels (S/R) accompany every reported number.
"""
