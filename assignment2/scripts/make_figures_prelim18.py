# [과제2] 새로 작성: 18개 소스 예비 결과 그림(참고용).
"""Presentation figures (PNG) for assignment 2: control vs time-compressed source.

Usage: python make_figures.py <control_prefix> <treatment_prefix> <ctrl_mimic.hdf5> <treat_mimic.hdf5> <out_dir>
  prefixes name rollout folders logs/assignment2/rollouts_<prefix>_seed{101,111}
"""
import glob
import json
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

L = os.path.dirname(os.path.abspath(__file__))
CTRL, TREAT, CTRL_H5, TREAT_H5, OUT = sys.argv[1:6]
os.makedirs(OUT, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_CTRL, C_TREAT = "#eb6834", "#2a78d6"  # validated pair (categorical slots 2, 1)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})
for f in ["NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans KR"]:
    if any(f in x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

LABEL = {CTRL: "대조군 (원본 시범)", TREAT: "처치군 (시간 압축)"}
COLOR = {CTRL: C_CTRL, TREAT: C_TREAT}


def rollouts(prefix):
    rows, reach = [], []
    for s in (101, 111):
        d = os.path.join(L, f"rollouts_{prefix}_seed{s}")
        rows += json.load(open(os.path.join(d, "classification.json")))
        for f in sorted(glob.glob(os.path.join(d, "trial_*.npz"))):
            z = np.load(f)
            dist = np.linalg.norm(z["obs_eef_pos"] - z["obs_toy_truck_positions"], axis=1)
            hit = np.where(dist <= 0.046)[0]
            reach.append(int(hit[0]) if len(hit) else None)
    return rows, reach


data = {p: rollouts(p) for p in (CTRL, TREAT)}

# 1) outcome / first-failure stage shift (mean of the two training seeds, per 100 rollouts)
stages = ["success", "approach", "grasp", "carry", "place"]
names = ["성공", "접근 실패", "파지 실패", "운반 실패", "배치 실패"]
fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(len(stages))
w = 0.38
for i, p in enumerate((CTRL, TREAT)):
    rows = data[p][0]
    v = [sum(r["stage"] == s for r in rows) / 2 for s in stages]
    bars = ax.bar(x + (i - 0.5) * (w + 0.02), v, w, color=COLOR[p], label=LABEL[p])
    for b, val in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, val + 1, f"{val:g}", ha="center", va="bottom", fontsize=11, color=INK)
ax.set_xticks(x, names)
ax.set_ylabel("100회 중 횟수 (seed 101·111 평균)")
ax.set_title("결과 유형 분포: 접근 실패가 줄고 병목이 배치로 이동", loc="left", fontsize=14)
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_outcome_stage_shift.png"), dpi=200)
plt.close(fig)

# 2) time to reach the truck: cumulative share of rollouts that reached it by step t
fig, ax = plt.subplots(figsize=(9, 4.8))
t = np.arange(0, 801)
for p in (CTRL, TREAT):
    r = [v for v in data[p][1]]
    n = len(r)
    cum = [sum(1 for v in r if v is not None and v <= k) / n * 100 for k in t]
    ax.plot(t, cum, color=COLOR[p], lw=2, label=LABEL[p])
    reached = [v for v in r if v is not None]
    med = float(np.median(reached))
    ax.text(805, cum[-1], f"{cum[-1]:.0f}%", va="center", fontsize=11, color=INK)
    ax.plot([med], [sum(1 for v in reached if v <= med) / n * 100], "o", ms=8, color=COLOR[p], mec=SURFACE, mew=2)
ax.axvline(800, color=INK2, lw=1, ls="--")
ax.text(795, 3, "평가 한도 800 step", ha="right", fontsize=10, color=INK2)
ax.set_xlim(0, 860)
ax.set_ylim(0, 100)
ax.set_xlabel("step")
ax.set_ylabel("트럭에 도달한 rollout 누적 비율 (%)")
ax.set_title("트럭 도달 시점: 처치군이 더 빨리, 더 많이 도달 (점 = 도달 시점 중앙값)", loc="left", fontsize=14)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_time_to_reach.png"), dpi=200)
plt.close(fig)

# 3) training data (Mimic) episode length vs the 800-step evaluation horizon
fig, ax = plt.subplots(figsize=(9, 4.8))
bins = np.arange(0, 2600, 50)
for p, h in ((CTRL, CTRL_H5), (TREAT, TREAT_H5)):
    with h5py.File(h, "r") as f:
        lens = np.array([f["data"][k]["actions"].shape[0] for k in f["data"]])
    over = (lens > 800).mean() * 100
    ax.hist(lens, bins=bins, histtype="step", color=COLOR[p], linewidth=2,
            label=f"{LABEL[p]}: 평균 {lens.mean():.0f} step, 800 초과 {over:.0f}%")
ax.axvline(800, color=INK, lw=1.5, ls="--")
ax.annotate("평가 한도 800 step", xy=(800, 105), xytext=(1130, 105), va="center", fontsize=10, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1))
ax.set_xlabel("학습 데이터(Mimic 500개) 에피소드 길이 (step)")
ax.set_ylabel("에피소드 수")
ax.set_title("학습 데이터 길이와 평가 시간 예산", loc="left", fontsize=14)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_episode_length_vs_horizon.png"), dpi=200)
plt.close(fig)
print("saved to", OUT)
