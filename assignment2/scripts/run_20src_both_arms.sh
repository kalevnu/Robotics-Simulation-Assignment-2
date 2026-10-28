#!/usr/bin/env bash
# [과제2] 새로 작성: 대조·처치 두 조건 생성→학습→평가 자동 실행(서버 경로 기준).
# P0: control vs time-compressed source, both from the same 20-demo annotation (2 rule-marked grasp boundaries).
# annotate (keep-no-signal copy, already started separately) -> rule boundary -> compressed copy
# -> 2 Mimic generations in parallel (own TMPDIRs) -> trainings 2 at a time -> 4 logged plays -> classification.
set -u
export PYTHONUNBUFFERED=1
cd ~/IsaacLab_RS
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena

L=logs/assignment2
TASK=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
MIMIC_TASK=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0
EXPROOT=./logs/robomimic/$TASK
note() { echo "[$(date '+%F %T')] $*" | tee -a $L/src20_pipeline.log; }
CUDNN=$L/cudnn_cu12/ext/nvidia/cudnn/lib
export LD_LIBRARY_PATH=$PWD/$CUDNN:$LD_LIBRARY_PATH
CUDNN_PRELOAD=$(ls $PWD/$CUDNN/libcudnn*.so.9 | tr '\n' ':')

RAW=./datasets/team08_inwoo_all20_annotated_v3_raw.hdf5
ANN=./datasets/team08_inwoo_all20_annotated_20src.hdf5
ANN_TC=./datasets/team08_inwoo_all20_annotated_20src_tcsrc.hdf5
declare -A DATA=([ctrl20]=./datasets/team08_inwoo_20src_mimic_v1_500.hdf5 [tcsrc20]=./datasets/team08_inwoo_20src_tcsrc_mimic_v1_500.hdf5)
declare -A SRC=([ctrl20]=$ANN [tcsrc20]=$ANN_TC)

# 1) wait for the keep-no-signal annotate run
while pgrep -f "annotate_keep_nosignal.py" >/dev/null; do sleep 30; done
N=$(python -I -c "import h5py;print(len(h5py.File('$RAW','r')['data']))")
note "annotate v3 raw: $N episodes"
[ "$N" = 20 ] || { note "ERROR: expected 20 episodes, abort"; exit 1; }

# 2) rule boundary for episodes without auto grasp signal, then compressed copy (same rule as tcsrc)
python -I $L/set_rule_grasp_boundary.py $RAW $ANN --dist 0.06 2>&1 | tee -a $L/src20_pipeline.log
python -I $L/time_transform_hdf5.py compress $ANN $ANN_TC --protect_signals 2>&1 | grep -E 'mean|share' | tee -a $L/src20_pipeline.log
[ -s $ANN_TC ] || { note "ERROR: compressed file missing"; exit 1; }

# 3) two generations in parallel
for A in ctrl20 tcsrc20; do [ -e ${DATA[$A]} ] && { note "ERROR: ${DATA[$A]} exists, abort"; exit 1; }; done
declare -A GPID
for A in ctrl20 tcsrc20; do
  mkdir -p $L/tmp_gen_$A
  note "generate $A start (source ${SRC[$A]})"
  TMPDIR=$PWD/$L/tmp_gen_$A ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task $MIMIC_TASK --headless --num_envs 100 --generation_num_trials 500 \
    --input_file ${SRC[$A]} --output_file ${DATA[$A]} > $L/gen_$A.log 2>&1 &
  GPID[$A]=$!
  sleep 60
done
for A in ctrl20 tcsrc20; do
  wait ${GPID[$A]}
  note "generate $A exit=$?: $(grep -a 'successful demos' $L/gen_$A.log | tail -1)"
  note "$A data: $(python -I $L/check_dup_initial_states.py ${DATA[$A]})"
done

# 4) trainings, 2 at a time (seed 101 pair, then seed 111 pair)
train_pair() {
  local S=$1; declare -A P
  for A in ctrl20 tcsrc20; do
    local NM=${A}_seed$S
    [ -e $EXPROOT/$NM ] && { note "ERROR: $EXPROOT/$NM exists, skip"; continue; }
    note "train $NM start"
    LD_PRELOAD=$CUDNN_PRELOAD ./isaaclab.sh -p $L/train_seeded.py \
      --task $TASK --algo bc --dataset ${DATA[$A]} --name $NM --seed $S > $L/train_$NM.log 2>&1 &
    P[$A]=$!
    sleep 30
  done
  for A in ctrl20 tcsrc20; do
    [ -n "${P[$A]:-}" ] && wait ${P[$A]}
    note "train ${A}_seed$S finished (epochs: $(grep -a -c 'Train Epoch' $L/train_${A}_seed$S.log), errors: $(grep -a -c -E 'Traceback|run failed' $L/train_${A}_seed$S.log))"
  done
}
logged_play() {
  local NM=$1
  local CK=$(ls -d $EXPROOT/$NM/*/models/model_epoch_2000.pth 2>/dev/null | tail -1)
  [ -z "$CK" ] && { note "ERROR: no checkpoint for $NM"; return; }
  mkdir -p $L/tmp_play_$NM
  note "play $NM start ckpt=$CK"
  TMPDIR=$PWD/$L/tmp_play_$NM LD_PRELOAD=$CUDNN_PRELOAD ./isaaclab.sh -p $L/play_logged.py \
    --task $TASK --checkpoint $CK --num_rollouts 100 --horizon 800 --seed 42 --headless \
    --rollout_log_dir $L/rollouts_$NM > $L/play_$NM.log 2>&1
  note "play $NM done: $(python -I -c "import json;d=json.load(open('$L/rollouts_$NM/summary.json'));print(d['successes'],'/',d['trials'])" 2>&1)"
}
train_pair 101
logged_play ctrl20_seed101 &
logged_play tcsrc20_seed101 &
train_pair 111
logged_play ctrl20_seed111 &
logged_play tcsrc20_seed111 &
wait
python -I $L/classify_rollouts.py $L/rollouts_ctrl20_seed101 $L/rollouts_ctrl20_seed111 \
  $L/rollouts_tcsrc20_seed101 $L/rollouts_tcsrc20_seed111 > $L/classification_20src.txt 2>&1
note "pipeline done (classification: $L/classification_20src.txt)"
