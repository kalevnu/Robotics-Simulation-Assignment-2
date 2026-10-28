# [과제2] 새로 작성: 기록된 rollout을 접근/파지/운반/배치 실패 단계와 행동 지표로 분류.
"""Classify logged rollouts (play_logged.py npz) into first-failure stage and per-rollout behaviour metrics.

Stage of first failure (thresholds fixed before looking at results):
  approach : EEF-truck distance never <= APPROACH_DIST (95th pct of EEF-truck distance at the grasp signal
             in our 20 demos = 0.046 m)
  grasp    : approached, but the truck was never held (truck z >= start + 0.02 m with EEF-truck distance <= 0.08 m
             for 10 consecutive steps)
  carry    : held, but truck-box xy distance never < 0.10 m
  place    : reached truck-box xy < 0.10 m but the success condition was not met
Behaviour metrics: steps, timeout (800 steps without success), idle ratio (EEF moves < 0.5 mm and rotates < 0.2 deg
in a step), stalls (idle runs >= 40 steps), gripper toggles (sign changes of the gripper command), oscillation
(sign changes per 100 steps of each arm action axis, ignoring |a| < 0.002).
Usage: python classify_rollouts.py <rollout_dir> [<rollout_dir> ...]
"""
import glob
import json
import os
import sys

import numpy as np

APPROACH_DIST = 0.046
LIFT = 0.02
HOLD_DIST = 0.08
HOLD_STEPS = 10
BOX_XY = 0.10
IDLE_POS = 5e-4
IDLE_ROT = np.deg2rad(0.2)
STALL = 40
OSC_EPS = 0.002


def runs(mask):
    """Lengths of consecutive True runs."""
    out, n = [], 0
    for m in mask:
        if m:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def quat_angle(q1, q2):
    dot = np.abs(np.sum(q1 * q2, axis=1)).clip(0, 1)
    return 2 * np.arccos(dot)


def classify(path):
    d = np.load(path)
    a = d["actions"]
    eef, truck, box = d["obs_eef_pos"], d["obs_toy_truck_positions"], d["obs_box_positions"]
    n = len(a)
    success = bool(d["success"])
    dist = np.linalg.norm(eef - truck, axis=1)
    held_mask = (truck[:, 2] >= truck[0, 2] + LIFT) & (dist <= HOLD_DIST)
    held = max(runs(held_mask), default=0) >= HOLD_STEPS
    box_xy = np.linalg.norm(truck[:, :2] - box[:, :2], axis=1)
    if success:
        stage = "success"
    elif dist.min() > APPROACH_DIST:
        stage = "approach"
    elif not held:
        stage = "grasp"
    elif box_xy.min() >= BOX_XY:
        stage = "carry"
    else:
        stage = "place"
    step_pos = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    step_rot = quat_angle(d["obs_eef_quat"][1:], d["obs_eef_quat"][:-1])
    idle = (step_pos < IDLE_POS) & (step_rot < IDLE_ROT)
    stalls = [r for r in runs(idle) if r >= STALL]
    grip = np.sign(a[:, 6])
    toggles = int(np.sum(grip[1:] != grip[:-1]))
    osc = []
    for ax in range(6):
        s = np.sign(np.where(np.abs(a[:, ax]) < OSC_EPS, 0, a[:, ax]))
        s = s[s != 0]
        osc.append(float(np.sum(s[1:] != s[:-1]) * 100 / n))
    return dict(
        trial=os.path.basename(path)[:-4], success=success, stage=stage, steps=n,
        timeout=(not success and n >= 800), min_eef_truck=float(dist.min()), held=bool(held),
        min_truck_box_xy=float(box_xy.min()), idle_ratio=float(idle.mean()), n_stalls=len(stalls),
        max_idle_run=max(runs(idle), default=0), grip_toggles=toggles, osc_per100=osc,
        osc_mean=float(np.mean(osc)),
    )


def summarize(rows):
    k = len(rows)
    fails = [r for r in rows if not r["success"]]
    out = dict(n=k, success=sum(r["success"] for r in rows))
    out["stage_counts"] = {s: sum(r["stage"] == s for r in rows) for s in ["approach", "grasp", "carry", "place"]}
    out["timeouts"] = sum(r["timeout"] for r in rows)
    for grp, rr in [("success", [r for r in rows if r["success"]]), ("fail", fails)]:
        if rr:
            out[grp] = dict(
                n=len(rr), steps_mean=float(np.mean([r["steps"] for r in rr])),
                idle_ratio_mean=float(np.mean([r["idle_ratio"] for r in rr])),
                frac_with_stall=float(np.mean([r["n_stalls"] > 0 for r in rr])),
                grip_toggles_mean=float(np.mean([r["grip_toggles"] for r in rr])),
                frac_toggles_ge3=float(np.mean([r["grip_toggles"] >= 3 for r in rr])),
                osc_mean=float(np.mean([r["osc_mean"] for r in rr])),
            )
    return out


if __name__ == "__main__":
    for rdir in sys.argv[1:]:
        rows = [classify(p) for p in sorted(glob.glob(os.path.join(rdir, "trial_*.npz")))]
        with open(os.path.join(rdir, "classification.json"), "w") as f:
            json.dump(rows, f, indent=1)
        print(os.path.basename(os.path.normpath(rdir)), json.dumps(summarize(rows), indent=1))
