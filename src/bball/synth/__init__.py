"""SYNTH — the synthetic-data experiment engine (Stage A's workhorse).

physics.py    3D ballistic shot generator (grounded in real basketball ranges)
camera.py     pinhole projection (azimuth/height/distance/FOV, iPhone-wide defaults)
render.py     motion-blurred ball over court backgrounds + detection-noise model
scenarios.py  scene/venue bundles + session generator; multi-agent player motion

Outputs BOTH ground-truth trajectories/events (logic-level experiments) and rendered
mp4 clips (end-to-end demos). Every physical constant is cited (plan gate G4).
"""
