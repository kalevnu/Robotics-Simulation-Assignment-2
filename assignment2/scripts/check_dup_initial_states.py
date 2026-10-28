# [과제2] 새로 작성: Mimic 생성 데이터의 초기 상태 중복 검사.
"""Count duplicate episodes in a Mimic hdf5 by comparing truck/box initial root poses."""
import sys

import h5py
import numpy as np

f = h5py.File(sys.argv[1], "r")
keys = list(f["data"].keys())
poses = np.array([
    np.concatenate([
        f["data"][k]["initial_state/rigid_object/toy_truck/root_pose"][0],
        f["data"][k]["initial_state/rigid_object/box/root_pose"][0],
    ])
    for k in keys
])
dup = 0
for i in range(len(poses)):
    if np.any(np.all(np.abs(poses[:i] - poses[i]) < 1e-6, axis=1)):
        dup += 1
print(f"episodes={len(keys)} duplicate_initial_states={dup}")
