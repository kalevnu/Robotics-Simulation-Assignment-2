# [과제2] 새로 작성: grasp 신호가 없는 시범 2개에 규칙(거리 0.06 m)으로 grasp 경계를 지정.
"""Copy an annotated hdf5 and set the grasp subtask boundary by rule only for episodes whose auto grasp signal is all False.

Rule (fixed before any results, validated against the auto signal on 18 demos: median -2 steps, 16/18 within 3 steps):
inside the first gripper-closed interval during which the truck is lifted >= 0.02 m above its start height, the first step
whose EEF-truck distance <= D (D = 0.06 m; the auto detector uses 0.05 m plus a two-finger contact force check).
The signal is False before that step and True from it on, the same format as annotate_demos manual mode.
Usage: python set_rule_grasp_boundary.py <in_annotated.hdf5> <out.hdf5> [--dist 0.06]
"""
import argparse
import os
import shutil

import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("input")
p.add_argument("output")
p.add_argument("--dist", type=float, default=0.06)
args = p.parse_args()
assert not os.path.exists(args.output), f"{args.output} exists"
shutil.copyfile(args.input, args.output)


def rule_step(a, truck, eef, d):
    z0 = truck[0, 2]
    closed = a[:, 6] < 0
    dist = np.linalg.norm(eef - truck, axis=1)
    t = 0
    while t < len(a):
        if closed[t] and (t == 0 or not closed[t - 1]):
            e = t
            while e < len(a) and closed[e]:
                e += 1
            if (truck[t:e, 2] >= z0 + 0.02).any():
                idx = np.where(dist[t:e] <= d)[0]
                return t + int(idx[0]) if len(idx) else None, float(dist[t:e].min())
            t = e
        else:
            t += 1
    return None, None


with h5py.File(args.output, "r+") as f:
    for k in sorted(f["data"], key=lambda s: int(s.split("_")[-1])):
        g = f["data"][k]
        sig = g["obs/datagen_info/subtask_term_signals/grasp"]
        if np.asarray(sig[:]).any():
            continue
        step, mind = rule_step(g["actions"][:], g["obs/toy_truck_positions"][:], g["obs/eef_pos"][:], args.dist)
        assert step is not None, f"{k}: rule found no boundary (min dist {mind})"
        new = np.ones(sig.shape, dtype=sig.dtype)
        new[:step] = False
        sig[...] = new
        print(f"{k}: rule grasp boundary at step {step} of {len(new)} (min EEF-truck distance in lift interval {mind:.3f} m)")
    print("episodes:", len(f["data"]), "all have a grasp signal:",
          all(np.asarray(f["data"][k]["obs/datagen_info/subtask_term_signals/grasp"][:]).any() for k in f["data"]))
