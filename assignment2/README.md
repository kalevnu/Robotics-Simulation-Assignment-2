# 실습과제 #2 — Agibot Toy2Box 모방학습 성공률 향상 연구 (8조)

> 고정 연구 질문: 사람 시범으로 학습한 BC 정책은 Toy2Box에서 왜 성공률이 저조한가? 원인 하나를 진단하고, 그 원인을 겨냥해 성공률을 높인다.

> **용어**
> - **step**: 시뮬레이션 제어 한 번(20 Hz, 1 step = 0.05초). 800 step = 40초.
> - **horizon**: 평가 한 번에 허용되는 최대 step 수. 과제 규칙상 800으로 고정.
> - **idle**: 시범에서 이동 키를 누르지 않은 step(팔 6축 action이 모두 0이고 그리퍼 명령도 그대로).
> - **Mimic**: 사람 시범 20개를 물체 위치에 맞게 변형·재조합해 새 시범 500개를 자동으로 만드는 Isaac Lab 도구.
> - **생성 수용률**: Mimic이 시도한 시범 중 성공 판정을 통과해 저장된 비율(예: 500/1,610 = 31.1%).
> - **seed**: 난수 시작값. 학습 seed 101·111(모델 2개), 평가 seed 42(초기 위치 순서).
> - **대조군/처치군**: 시범 원본으로 학습한 모델 / 시간 압축 시범으로 학습한 모델. 그 외 조건은 모두 같다.

**한 줄 요약**: 키보드 원격조작 시범은 느려서(평균 1,083 step) 평가 시간 예산(800 step)을 넘고, BC 정책은 그 속도를 그대로 따라 해 트럭에 닿는 데만 시간을 다 쓴다(시간예산 결손). 같은 시범 20개의 시간축만 압축하자 성공률이 **3.0% → 27.0%**로 올랐다.

## 1. Baseline 재현과 진단

| 데이터 (사람 시범 → Mimic 성공 500) | seed 101 | seed 111 | 평균 ± 표준편차 |
|---|---|---|---|
| 조교 시범 20개 (조교 config 재현) | 35/100 | 44/100 | 39.5 ± 6.4% (조교 공지 26/100) |
| 우리 시범 20개 (대조군) | 3/100 | 3/100 | 3.0 ± 0.0% |

평가: 검증 seed 42, rollout 100회, horizon 800, epoch 2000 체크포인트. 조교 공지 26%와 우리 재현 39.5%의 차이 원인은 확인하지 않았다.

**실패 진단** (실패 단계 = 처음 달성하지 못한 단계):
- 우리 정책 실패의 대부분이 접근 단계(100회 중 51·67회)이고, 95% 이상이 800 step 시간 초과.
- 우리 시범은 평균 1,083 step으로 80%가 800 step을 넘는다(조교 시범 545 step, 5%). 시범의 멈춤 비율 49%(강의자료 권장 30% 이하, 조교 28%).
- 정책은 소스 데이터의 속도를 그대로 재현한다: 트럭 도달 중앙값 515·589 step(우리), 373 step(조교).
- **기각한 가설**: "정책이 사람의 멈춤을 따라 해 정체한다" — 정책의 정지 비율 0–5%, 40 step 이상 정체 0회.
- 조교 정책도 실패의 거의 전부가 시간 초과(65/65, 55/56)로, 같은 기제가 더 약하게 나타난다(트럭 도달 후 운반·배치 중 시간 초과).

## 2. 가설과 처치 (연구 축 A: 데이터)

**가설**: 평가 시간 예산(800 step) 안에서 접근에만 500 step 이상을 쓰는 것이 저조한 성공률의 원인이다. 시범의 시간축을 압축하면 정책이 트럭에 더 빨리·더 자주 도달하고 성공률이 오른다.

**처치 (바꾸는 요인 1개: 시범의 시간축)** — `assignment2/scripts/time_transform_hdf5.py compress --protect_signals`
- annotate된 시범에서 idle step(팔 6축 action이 모두 0이고 그리퍼 명령이 그대로인 step)의 연속 구간을 **최대 3 step**만 남긴다.
- 그리퍼 전환 전후 각 5 step, grasp 경계 전후 5 step, 마지막 10 step(성공 판정 대기)은 항상 유지한다.
- 같은 압축 규칙을 사람 시범 파일에 적용해 그대로 재생(replay)하면 16/20만 성공(idle 5 step 허용 시 15/20)해서, 압축은 Mimic 소스(annotate된 파일) 층에 적용했다(압축 소스 자체를 그대로 재생해도 15/20이다, 4절 참조). Mimic은 물체 기준 목표 자세를 시뮬레이션에서 다시 실행하며, 생성된 시범은 각각 성공 판정을 통과해야 저장된다.
- 길이: 평균 1,083 → 666 step, 800 step 초과 80% → 5%.

**고정 조건**: 같은 사람 시범 20개, Mimic 성공 500개(noise 0.01, 보간 15), 조교 BC-RNN config(LSTM 2×400, GMM 5, batch 100, 2,000 epoch), 학습 seed 101·111, 평가 seed 42·100회·horizon 800. 환경·성공 판정·종료 조건은 바꾸지 않았다.

**grasp 경계 규칙 지정 2개**: 우리 시범 중 2개(원본 demo_11, demo_19)는 트럭을 손 중심에서 약간 벗어나게 잡아(들어 올린 구간의 손–트럭 최소 거리 0.054 m, 0.057 m) 자동 grasp 감지(0.05 m + 접촉력)가 켜지지 않았다. 이 2개는 "트럭이 2 cm 이상 들리는 그리퍼 닫힘 구간에서 손–트럭 거리 ≤ 0.06 m가 되는 첫 step"을 경계로 지정했다(수동 annotate 모드와 같은 형식). 이 규칙은 나머지 18개에서 자동 경계와 중앙값 −2 step 차이(0.05 m 기준 16/18이 3 step 이내)였다. 대조군과 처치군은 같은 20개 annotate 파일을 쓴다.

## 3. 결과 (사전 판정 기준: 평균 ≥ 14%, 각 seed ≥ 9%, 도달 중앙값 ≤ 446 step 또는 도달률 ≥ 57%)

| 지표 | 대조 s101 | 대조 s111 | 처치 s101 | 처치 s111 |
|---|---|---|---|---|
| 성공 /100 | 3 | 3 | **23** | **31** |
| 트럭 도달 /100 | 49 | 33 | 75 | 87 |
| 도달 step 중앙값 (도달한 rollout만) | 515 | 589 | 442 | 422 |
| 실패 단계 접근/파지/운반/배치 | 51/20/13/13 | 67/14/13/3 | 24/16/27/10 | 13/16/30/10 |
| 800 step 시간 초과 | 94 | 97 | 75 | 65 |

- 성공률 **3.0 ± 0.0% → 27.0 ± 5.7%**, 두 학습 seed 모두 개선 → 사전 기준상 **가설 지지**.
- Mimic 생성 수용률: 대조 29.4%, 처치 31.1%.
- 접근 실패가 크게 줄고(평균 59 → 18.5), 병목이 운반·배치 단계로 이동했다(운반 실패 13 → 28.5). 시간 초과는 여전히 65–75%.
- 그림: `assignment2/figures/fig_a_success_rate.png`, `assignment2/figures/fig_b_outcome_stages.png`, `assignment2/figures/fig_c_length_and_reach.png`

추가 검증 (작성 예정):
- 추가 평가 seed 7: (P1 결과 기입)
- 실패 분류기 수동 검증 영상 30개: (P2 결과 기입)
- 진단용 horizon 1,600 평가(공식 결과 아님): (P5 결과 기입)

## 4. 한계
- 압축된 소스 시범을 단일 환경에서 그대로 재생하면 15/20만 성공한다(원본 20/20). 생성 데이터의 물리적 타당성은 Mimic 생성 수용률과 생성 시범 각각의 성공 판정에만 근거한다.
- 대조·처치는 같은 평가 seed 42를 쓰지만 rollout 길이가 달라 몇 번째 trial부터 초기 상태가 갈라진다(짝지은 비교가 아님).
- 학습 seed 2개, 평가 seed 1개(추가 seed 결과는 위 표에 보충).
- grasp 경계 2개는 규칙으로 지정했고 영상으로 확인하지 않았다(관측값으로 파지→들기 전환점임을 확인).
- 시범 18개 소스로 먼저 한 예비 비교(4.0% → 24.0%)는 "사람 시범 최소 20개" 규칙에 못 미쳐 탐색 결과로만 사용했다.
- 실험 서버의 conda 환경에 CUDA 13용 cuDNN이 덮어써져 있어, 학습·평가 때만 CUDA 12용 cuDNN 9.7.1을 preload했다(우리 환경 전용 우회, 정상 환경에서는 불필요).

## 5. 재현 방법 (저장소 루트, conda env `lerobot-arena`)
1. annotate (grasp 신호가 없는 시범도 내보내는 사본): `assignment2/scripts/annotate_keep_nosignal.py --auto --input_file assignment2/datasets/team08_inwoo_all20.hdf5 ...`
2. 규칙 경계 지정: `python assignment2/scripts/set_rule_grasp_boundary.py <annotated_raw> <annotated_20src> --dist 0.06` (결과물: `assignment2/datasets/team08_inwoo_all20_annotated_20src.hdf5`)
3. 처치 소스 만들기: `python assignment2/scripts/time_transform_hdf5.py compress <annotated_20src> <annotated_20src_tcsrc> --protect_signals` (결과물: `assignment2/datasets/team08_inwoo_all20_annotated_20src_tcsrc.hdf5`)
4. Mimic 생성(조교 명령과 동일, 성공 500): `scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 --headless --num_envs 100 --generation_num_trials 500 ...`
5. 학습(조교 config, seed만 지정): `assignment2/scripts/train_seeded.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --algo bc --dataset <mimic500> --name <이름> --seed 101|111`
6. 평가: `assignment2/play_commands.txt` (기본 play.py). 실패 분류는 `assignment2/scripts/play_logged.py`(play.py에 rollout 기록만 추가) + `assignment2/scripts/classify_rollouts.py`.
- 전체 자동화 예: `assignment2/scripts/run_20src_both_arms.sh` (서버 경로 기준이라 경로 수정 필요).
- 병렬 실행 시 프로세스마다 `TMPDIR`를 따로 지정해야 한다(RmpFlow가 매 reset마다 URDF를 임시 폴더에 다시 받음).

## 6. 포함 파일 (`assignment2/` 기준 경로)
- `weights/seed101/model_epoch_2000.pth`, `weights/seed111/model_epoch_2000.pth` (+ 각 `config.json`): 최종 모델(처치군)
- `datasets/team08_inwoo_all20.hdf5`: 우리 사람 시범 20개(키보드 수집 원본)
- `datasets/team08_inwoo_all20_annotated_20src.hdf5`: annotate 20개(대조군 Mimic 소스)
- `datasets/team08_inwoo_all20_annotated_20src_tcsrc.hdf5`: 시간 압축 20개(처치군 Mimic 소스)
- Mimic 생성 데이터(각 274–383 MB)는 용량 때문에 제외했다(위 4단계로 재생성).
- `scripts/`: 우리가 추가한 스크립트(조교 원본 스크립트는 수정하지 않았고 필요한 것은 사본으로 만들었다).
