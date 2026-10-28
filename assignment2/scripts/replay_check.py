# [과제2] 새로 작성: 시범을 그대로 재생해 성공 판정(10 step 연속)을 확인.
"""Replay demos and report whether the task success term holds (read-only on dataset)."""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--dataset_file", required=True)
p.add_argument("--select_episodes", type=int, nargs="+", default=[])
p.add_argument("--video_folder", default=None)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.video_folder:
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym, torch, os
from isaaclab.utils.datasets import HDF5DatasetFileHandler
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

h = HDF5DatasetFileHandler(); h.open(args.dataset_file)
names = sorted(h.get_episode_names(), key=lambda k: int(k.split("_")[-1]))  # numeric order: demo_0, demo_1, ..., demo_19
idx = args.select_episodes or list(range(len(names)))
cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
succ = cfg.terminations.success
cfg.recorders = {}; cfg.terminations = {}
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video_folder else None)
if args.video_folder:
    env = gym.wrappers.RecordVideo(env, args.video_folder, episode_trigger=lambda e: True, disable_logger=True,
                                   name_prefix="replay")
u = env.unwrapped
if not args.video_folder:
    env.reset()
results = []
with torch.inference_mode():
    for i in idx:
        ep = h.load_episode(names[i], u.device)
        if args.video_folder:
            env.reset()  # start a new video segment
        u.reset_to(ep.get_initial_state(), torch.tensor([0], device=u.device), is_relative=True)
        run = best = 0; first = None; n = 0
        while (a := ep.get_next_action()) is not None:
            env.step(a.unsqueeze(0)); n += 1
            ok = bool(succ.func(u, **succ.params)[0])
            run = run + 1 if ok else 0; best = max(best, run)
            if run >= 10 and first is None: first = n
        results.append((i, n, best, first))
        print(f"[CHECK] {names[i]}: steps={n} max_consecutive_success={best} success10_at={first} -> {'SUCCESS' if first else 'FAIL'}", flush=True)
print(f"[CHECK] replay success {sum(r[3] is not None for r in results)}/{len(results)}", flush=True)
env.close(); app.close()
