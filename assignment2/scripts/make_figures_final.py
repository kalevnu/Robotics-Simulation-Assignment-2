# [과제2] 새로 작성: 최종 발표 그림 3장(20개 소스 결과).
"""Final presentation figures (P0, 20 sources): success per seed, outcome-stage shift, reach time vs source length.
Usage: python make_figures_final.py <out_dir>
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
D = os.path.join(L, "..", "..", "datasets")
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_CTRL, C_TREAT = "#eb6834", "#2a78d6"  # validated pair (categorical slots 2, 1)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.family": "Noto Sans CJK JP", "axes.unicode_minus": False,
})
ARMS = [("ctrl20", "대조군 (원본 시범)", C_CTRL), ("tcsrc20", "처치군 (시간 압축)", C_TREAT)]


def load(prefix, seed):
    d = os.path.join(L, f"rollouts_{prefix}_seed{seed}")
    rows = json.load(open(os.path.join(d, "classification.json")))
    reach = []
    for f in sorted(glob.glob(os.path.join(d, "trial_*.npz"))):
        z = np.load(f)
        dist = np.linalg.norm(z["obs_eef_pos"] - z["obs_toy_truck_positions"], axis=1)
        hit = np.where(dist <= 0.046)[0]
        reach.append(int(hit[0]) if len(hit) else None)
    return rows, reach


data = {(p, s): load(p, s) for p, _, _ in ARMS for s in (101, 111)}

# (a) success rate per seed + mean ± SD, with TA reference lines
fig, ax = plt.subplots(figsize=(9, 4.8))
xs = {"ctrl20": 0, "tcsrc20": 1}
for p, lab, col in ARMS:
    v = [sum(r["success"] for r in data[(p, s)][0]) for s in (101, 111)]
    m, sd = np.mean(v), np.std(v, ddof=1)
    ax.bar(xs[p], m, 0.5, color=col)
    ax.errorbar(xs[p], m, yerr=sd, color=INK, capsize=6, lw=1.5)
    for dx, val, sname in zip((-0.12, 0.12), v, ("101", "111")):
        ax.plot(xs[p] + dx, val, "o", ms=8, color=SURFACE, mec=INK, mew=1.5)
    ax.text(xs[p], m + sd + 2, f"{m:.1f} ± {sd:.1f}%", ha="center", fontsize=12, color=INK)
for y, t in ((26, "조교 공지 baseline 26%"), (39.5, "조교 데이터 재현 39.5%")):
    ax.axhline(y, color=INK2, lw=1, ls="--")
    ax.text(-0.55, y + 0.8, t, ha="left", fontsize=10, color=INK2)
ax.set_xticks([0, 1], [a[1] for a in ARMS])
ax.set_xlim(-0.6, 1.5)
ax.set_ylim(0, 50)
ax.set_ylabel("성공률 (%) · 평가 seed 42, 100회")
ax.set_title("우리 시범 20개: 시간 압축 전후 성공률 (점 = 학습 seed 101·111)", loc="left", fontsize=14)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_a_success_rate.png"), dpi=200)
plt.close(fig)

# (b) outcome / first-failure stage per arm (mean of 2 seeds), timeouts annotated
stages = ["success", "approach", "grasp", "carry", "place"]
names = ["성공", "접근 실패", "파지 실패", "운반 실패", "배치 실패"]
fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(len(stages))
w = 0.38
for i, (p, lab, col) in enumerate(ARMS):
    rows = data[(p, 101)][0] + data[(p, 111)][0]
    v = [sum(r["stage"] == s for r in rows) / 2 for s in stages]
    to = sum(r["timeout"] for r in rows) / 2
    bars = ax.bar(x + (i - 0.5) * (w + 0.02), v, w, color=col, label=f"{lab} · 800 step 시간 초과 {to:g}회")
    for b, val in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, val + 1, f"{val:g}", ha="center", va="bottom", fontsize=11, color=INK)
ax.set_xticks(x, names)
ax.set_ylabel("100회 중 횟수 (seed 101·111 평균)")
ax.set_title("결과 유형: 접근 실패가 줄고 병목이 운반·배치로 이동", loc="left", fontsize=14)
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_b_outcome_stages.png"), dpi=200)
plt.close(fig)

# (c) top: source demo length (20 demos) vs the 800-step budget; bottom: when the policy reaches the truck
src = {"ctrl20": "team08_inwoo_all20_annotated_20src.hdf5", "tcsrc20": "team08_inwoo_all20_annotated_20src_tcsrc.hdf5"}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
for row, (p, lab, col) in enumerate(ARMS):
    with h5py.File(os.path.join(D, src[p]), "r") as f:
        lens = np.array([f["data"][k]["actions"].shape[0] for k in f["data"]])
    y = np.full(len(lens), 1 - row) + np.linspace(-0.12, 0.12, len(lens))
    ax1.plot(lens, y, "o", ms=8, color=col, mec=SURFACE, mew=1.5, ls="none")
    ax1.text(2500, 1 - row + 0.32, f"평균 {lens.mean():.0f} step · 800 초과 {(lens > 800).mean() * 100:.0f}%",
             ha="right", va="center", fontsize=11, color=INK)
ax1.set_yticks([1, 0], [a[1] for a in ARMS])
ax1.set_ylim(-0.6, 1.6)
ax1.grid(axis="y", visible=False)
ax1.set_title("사람 시범 20개 길이 (위) 와 정책의 트럭 도달 시점 (아래)", loc="left", fontsize=14)
t = np.arange(0, 2501)
for p, lab, col in ARMS:
    r = data[(p, 101)][1] + data[(p, 111)][1]
    reached = [v for v in r if v is not None]
    cum = np.array([sum(1 for v in reached if v <= k) for k in t]) / len(r) * 100
    ax2.plot(t[:801], cum[:801], color=col, lw=2, label=lab)
    meds = [np.median([v for v in data[(p, s)][1] if v is not None]) for s in (101, 111)]
    ax2.text(810, cum[800], f"{cum[800]:.0f}% 도달 · 중앙값 {meds[0]:.0f}·{meds[1]:.0f} step", va="center", fontsize=11, color=INK)
for a in (ax1, ax2):
    a.axvline(800, color=INK, lw=1.5, ls="--")
ax1.annotate("평가 한도 800 step", xy=(800, 1.45), xytext=(1150, 1.45), va="center", fontsize=10, color=INK,
             arrowprops=dict(arrowstyle="->", color=INK, lw=1))
ax2.set_ylim(0, 100)
ax2.set_xlim(0, 2550)
ax2.set_xlabel("step (20 Hz)")
ax2.set_ylabel("트럭에 도달한 rollout 누적 비율 (%)")
ax2.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_c_length_and_reach.png"), dpi=200)
plt.close(fig)
print("saved to", OUT)
