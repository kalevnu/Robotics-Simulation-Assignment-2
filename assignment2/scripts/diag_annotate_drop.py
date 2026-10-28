# [과제2] 새로 작성: annotate에서 시범이 빠지는 이유 진단용 재생.
"""Diagnose why annotate_demos --auto drops some demos: replay them in the Mimic env exactly like annotate_demos.replay_episode
(env.sim.reset, recorder reset, reset_to(initial_state), step all actions, success at the last step), print the success
signal over the last steps, then continue with N extra hold steps (zero arm action, last gripper command) and report when
success first holds. Read-only on the dataset."""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0")
p.add_argument("--dataset_file", required=True)
p.add_argument("--demos", nargs="+", required=True)
p.add_argument("--extra", type=int, default=40)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

h = HDF5DatasetFileHandler()
h.open(args.dataset_file)
cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
succ = cfg.terminations.success
cfg.terminations.success = None
cfg.terminations.time_out = None
cfg.recorders = {}
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
with torch.inference_mode():
    for name in args.demos:
        ep = h.load_episode(name, env.device)
        actions = ep.data["actions"]
        env.sim.reset()
        env.reset_to(ep.data["initial_state"], None, is_relative=True)
        hist = []
        for a in actions:
            env.step(torch.as_tensor(a, device=env.device).reshape(1, -1).float())
            hist.append(int(bool(succ.func(env, **succ.params)[0])))
        last = actions[-1].clone()
        last[:6] = 0
        extra = []
        for _ in range(args.extra):
            env.step(last.reshape(1, -1).float())
            extra.append(int(bool(succ.func(env, **succ.params)[0])))
        print(f"[DIAG] {name}: steps={len(actions)} success_at_last_step={hist[-1]} "
              f"last30={''.join(map(str, hist[-30:]))} hold{args.extra}={''.join(map(str, extra))}", flush=True)
env.close()
app.close()
