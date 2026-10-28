#!/usr/bin/env bash
# [과제2] 새로 작성: 추가 평가 seed 7(P1)과 영상 재실행(P2).
# After P0: P1 = extra evaluation seed 7 (fixed before results) for the 4 20-source models (logged play + classification);
# P2 = video re-run of tcsrc20_seed101 at seed 42 (deterministic, same outcomes as the logged run) for the 30-video check.
set -u
export PYTHONUNBUFFERED=1
cd ~/IsaacLab_RS
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
L=logs/assignment2
TASK=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
EXPROOT=./logs/robomimic/$TASK
note() { echo "[$(date '+%F %T')] $*" | tee -a $L/p1_p2.log; }
CUDNN=$L/cudnn_cu12/ext/nvidia/cudnn/lib
export LD_LIBRARY_PATH=$PWD/$CUDNN:$LD_LIBRARY_PATH
CUDNN_PRELOAD=$(ls $PWD/$CUDNN/libcudnn*.so.9 | tr '\n' ':')

while ! grep -q 'pipeline done' $L/src20_pipeline.log; do sleep 60; done
note "P0 done; starting P1 (seed 7) and P2 (videos)"

run() {  # name seed outdir extra-args
  local NM=$1 SD=$2 OUT=$3; shift 3
  local CK=$(ls -d $EXPROOT/$NM/*/models/model_epoch_2000.pth | tail -1)
  [ -e $OUT ] && { note "ERROR: $OUT exists, skip"; return; }
  mkdir -p $L/tmp_p12_$(basename $OUT)
  note "play $NM seed $SD -> $OUT"
  TMPDIR=$PWD/$L/tmp_p12_$(basename $OUT) LD_PRELOAD=$CUDNN_PRELOAD ./isaaclab.sh -p $L/play_logged.py \
    --task $TASK --checkpoint $CK --num_rollouts 100 --horizon 800 --seed $SD --headless \
    --rollout_log_dir $OUT "$@" > $OUT.log 2>&1
  note "done $OUT: $(python -I -c "import json;d=json.load(open('$OUT/summary.json'));print(d['successes'],'/',d['trials'])" 2>&1)"
}
for NM in ctrl20_seed101 ctrl20_seed111 tcsrc20_seed101 tcsrc20_seed111; do
  run $NM 7 $L/rollouts_${NM}_eval7 &
done
run tcsrc20_seed101 42 $L/rollouts_tcsrc20_seed101_video --video --video_length 800 --video_folder $L/videos_tcsrc20_seed101 &
wait
python -I $L/classify_rollouts.py $L/rollouts_ctrl20_seed101_eval7 $L/rollouts_ctrl20_seed111_eval7 \
  $L/rollouts_tcsrc20_seed101_eval7 $L/rollouts_tcsrc20_seed111_eval7 > $L/classification_eval7.txt 2>&1
note "P1/P2 done"
