# 과제2 스크립트 (실행 순서)

수업 저장소의 원본 스크립트는 바꾸지 않았다. 바꿔야 할 때는 사본을 만들고 파일 첫 줄에 원본 경로를 적었다.
`run_*.sh`는 과제를 실행한 서버의 폴더 구성(`logs/assignment2/`) 기준이라, 다른 곳에서 쓰려면 경로를 고쳐야 한다.

| 순서 | 파일 | 종류 | 하는 일 |
|---|---|---|---|
| 1 | `replay_check.py` | 새로 작성 | 시범을 그대로 재생해 성공 판정(10 step 연속)을 확인 |
| 2 | `annotate_keep_nosignal.py` | 사본 (`annotate_demos.py`) | Mimic용 annotate. grasp 신호가 자동 감지되지 않은 시범도 내보냄 |
| 3 | `diag_annotate_drop.py` | 새로 작성 | annotate에서 시범이 빠지는 이유 진단 |
| 4 | `set_rule_grasp_boundary.py` | 새로 작성 | grasp 신호가 없는 시범 2개에 규칙으로 경계 지정 |
| 5 | `time_transform_hdf5.py` | 새로 작성 | 시범 시간축 압축 → 처치군 소스 |
| 6 | `generate_dataset_seeded.py` | 사본 (`generate_dataset.py`) | `--datagen_seed` 추가. 조교 데이터 생성 이어 하기에만 사용(우리 두 조건은 원본 스크립트 사용) |
| 7 | `check_dup_initial_states.py` | 새로 작성 | 생성 데이터의 초기 상태 중복 검사 |
| 8 | `train_seeded.py` | 사본 (`robomimic/train.py`) | `--seed` 추가(학습 seed 101·111) |
| 9 | `play_logged.py` | 사본 (`robomimic/play.py`) | 평가 로직 동일, rollout별 기록(npz, summary.json) 저장 |
| 10 | `classify_rollouts.py` | 새로 작성 | 실패 단계(접근/파지/운반/배치)와 행동 지표 분류 |
| 11 | `make_figures_final.py` | 새로 작성 | 최종 발표 그림 3장 |
| – | `make_figures_prelim18.py` | 새로 작성 | 18개 소스 예비 결과 그림(참고용) |
| – | `run_20src_both_arms.sh` | 새로 작성 | 4–10단계 자동 실행(대조·처치 두 조건) |
| – | `run_p1_p2.sh` | 새로 작성 | 추가 평가 seed 7, 실패 영상 재실행 |
| – | `run_p5_h1600.sh` | 새로 작성 | horizon 1,600 진단 평가(공식 결과 아님) |

병렬로 여러 시뮬레이션을 돌릴 때는 프로세스마다 `TMPDIR`를 따로 지정한다. RmpFlow가 reset마다 로봇 URDF를 임시 폴더에 다시 받아서, 같은 폴더를 쓰면 서로 덮어써 오류가 난다.
