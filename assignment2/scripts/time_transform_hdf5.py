# [과제2] 새로 작성: 시범 시간축 압축(idle 연속 최대 3 step, 그리퍼 전환·grasp 경계·마지막 10 step 보호). 처치군 소스 생성.
"""Time-axis transforms of an Isaac Lab demo hdf5 (writes a new file, never touches the input).

mode "compress" (A arm, human demos): an idle step = all 6 arm actions are 0 and the gripper command is unchanged.
  Inside each run of idle steps keep at most --max_idle steps, except that idle steps within --keep_around steps of a
  gripper toggle and the last --keep_tail steps of the demo are always kept.
mode "k2" (backup arm, Mimic data): merge consecutive step pairs (2j, 2j+1) into one step. Position deltas are summed,
  rotation deltas are composed as rotation vectors in the order the action term applies them (q2 * q1, world frame),
  the gripper takes the second value. Observations/states of the merged step are those of step 2j.
All per-step datasets (first dim == number of actions, outside initial_state) are subset with the same indices.
Usage: python time_transform_hdf5.py <mode> <in.hdf5> <out.hdf5> [--max_idle 3 --keep_around 5 --keep_tail 10] [--demos N]
"""
import argparse
import json
import os

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["compress", "k2"])
p.add_argument("input")
p.add_argument("output")
p.add_argument("--max_idle", type=int, default=3)
p.add_argument("--keep_around", type=int, default=5)
p.add_argument("--keep_tail", type=int, default=10)
p.add_argument("--protect_signals", action="store_true", help="also keep --keep_around steps around the first 0->1 transition of each obs/datagen_info/subtask_term_signals (annotated files)")
p.add_argument("--demos", type=int, default=None, help="only the first N demos (sorted by index)")
args = p.parse_args()
assert not os.path.exists(args.output), f"{args.output} exists"


def compress_indices(a, signal_steps=()):
    n = len(a)
    grip = np.sign(a[:, 6])
    idle = np.all(a[:, :6] == 0, axis=1) & np.r_[True, grip[1:] == grip[:-1]]
    protect = np.zeros(n, bool)
    for t in np.where(grip[1:] != grip[:-1])[0] + 1:
        protect[max(0, t - args.keep_around): t + args.keep_around + 1] = True
    for t in signal_steps:
        protect[max(0, t - args.keep_around): t + args.keep_around + 1] = True
    protect[max(0, n - args.keep_tail):] = True
    keep, run = [], 0
    for t in range(n):
        if idle[t] and not protect[t]:
            run += 1
            if run > args.max_idle:
                continue
        else:
            run = 0
        keep.append(t)
    return np.array(keep)


def k2_actions(a):
    n = len(a)
    out = []
    for j in range(0, n, 2):
        if j + 1 >= n:
            out.append(a[j])
            continue
        a1, a2 = a[j], a[j + 1]
        pos = a1[:3] + a2[:3]
        rot = (Rotation.from_rotvec(a2[3:6]) * Rotation.from_rotvec(a1[3:6])).as_rotvec()
        out.append(np.concatenate([pos, rot, a2[6:]]))
    return np.array(out, dtype=a.dtype)


def subset(src, dst, idx, n):
    for name, item in src.items():
        if name == "initial_state":
            src.copy(item, dst, name=name)
        elif isinstance(item, h5py.Group):
            g = dst.create_group(name)
            for k, v in item.attrs.items():
                g.attrs[k] = v
            subset(item, g, idx, n)
        elif item.shape and item.shape[0] == n:
            dst.create_dataset(name, data=item[:][idx], compression="gzip")
        else:
            src.copy(item, dst, name=name)


stats = []
with h5py.File(args.input, "r") as fi, h5py.File(args.output, "w") as fo:
    for k, v in fi.attrs.items():
        fo.attrs[k] = v
    di = fi["data"]
    do = fo.create_group("data")
    for k, v in di.attrs.items():
        do.attrs[k] = v
    names = sorted(di.keys(), key=lambda s: int(s.split("_")[-1]))
    if args.demos:
        names = names[: args.demos]
    total = 0
    for name in names:
        g = di[name]
        a = g["actions"][:]
        n = len(a)
        if args.mode == "compress":
            sig = []
            if args.protect_signals:
                for v in g["obs/datagen_info/subtask_term_signals"].values():
                    x = v[:].astype(int).reshape(-1)
                    sig += list(np.where(np.diff(x) > 0)[0] + 1)
            idx = compress_indices(a, sig)
            new_a = a[idx]
        else:
            idx = np.arange(0, n, 2)
            new_a = k2_actions(a)
        go = do.create_group(name)
        for kk, vv in g.attrs.items():
            go.attrs[kk] = vv
        subset(g, go, idx, n)
        del go["actions"]
        go.create_dataset("actions", data=new_a, compression="gzip")
        go.attrs["num_samples"] = len(new_a)
        total += len(new_a)
        stats.append((name, n, len(new_a)))
    do.attrs["total"] = total

before = np.array([s[1] for s in stats])
after = np.array([s[2] for s in stats])
summary = dict(
    mode=args.mode, params=vars(args), demos=len(stats),
    steps_before_mean=float(before.mean()), steps_after_mean=float(after.mean()),
    share_over_800_before=float((before > 800).mean()), share_over_800_after=float((after > 800).mean()),
    per_demo=[dict(demo=s[0], before=int(s[1]), after=int(s[2])) for s in stats],
)
with open(args.output + ".stats.json", "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "per_demo"}, indent=1))
