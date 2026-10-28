#!/usr/bin/env bash
# [과제2] 새로 작성: horizon 1,600 진단 평가(공식 결과 아님).
# P5 (diagnostic only, never an official result): the 4 P0 models evaluated with horizon 1600 (= 2 x 800, fixed before
# results), seed 42, 100 rollouts, logging play. Runs after P0, in parallel with P1/P2 (own TMPDIR per process).
set -u
export PYTHONUNBUFFERED=1
cd ~/IsaacLab_RS
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
L=logs/assignment2
TASK=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
EXPROOT=./logs/robomimic/$TASK
note() { echo "[$(date '+%F %T')] $*" | tee -a $L/p5_h1600.log; }
CUDNN=$L/cudnn_cu12/ext/nvidia/cudnn/lib
export LD_LIBRARY_PATH=$PWD/$CUDNN:$LD_LIBRARY_PATH
CUDNN_PRELOAD=$(ls $PWD/$CUDNN/libcudnn*.so.9 | tr '\n' ':')

while ! grep -q 'pipeline done' $L/src20_pipeline.log; do sleep 60; done
note "P0 done; starting P5 horizon 1600 (diagnostic only)"
run() {
  local NM=$1 OUT=$L/rollouts_h1600_$1
  local CK=$(ls -d $EXPROOT/$NM/*/models/model_epoch_2000.pth | tail -1)
  [ -e $OUT ] && { note "ERROR: $OUT exists, skip"; return; }
  mkdir -p $L/tmp_p5_$NM
  note "play $NM horizon 1600 seed 42"
  TMPDIR=$PWD/$L/tmp_p5_$NM LD_PRELOAD=$CUDNN_PRELOAD ./isaaclab.sh -p $L/play_logged.py \
    --task $TASK --checkpoint $CK --num_rollouts 100 --horizon 1600 --seed 42 --headless \
    --rollout_log_dir $OUT > $OUT.log 2>&1
  note "done $NM: $(python -I -c "import json;d=json.load(open('$OUT/summary.json'));print(d['successes'],'/',d['trials'])" 2>&1)"
}
for NM in ctrl20_seed101 ctrl20_seed111 tcsrc20_seed101 tcsrc20_seed111; do run $NM & done
wait
note "P5 done"
