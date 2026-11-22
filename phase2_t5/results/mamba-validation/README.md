# Mamba 검증 결과

Fashion-MNIST 공식 학습 이미지에서 공통 분할로 학습 48,000장과 검증 12,000장을 사용했다. 다섯 스캔에서 모델 seed 0·1·2 각각 20 epoch를 실행했다. 같은 14×14 anti-aliased 전처리와 스캔 규칙을 분류 입력과 신호 분석에 적용했다. 아래 값은 각 seed에서 검증 macro-F1이 가장 높은 체크포인트로 계산한 평균 ± 표본 표준편차다.

| 스캔 | 검증 Accuracy (%) | 검증 Macro-F1 (%) |
| --- | ---: | ---: |
| Axis-aligned (Raster) | 87.26 ± 0.08 | 87.22 ± 0.17 |
| Snake | 87.48 ± 0.29 | 87.50 ± 0.26 |
| Random | 86.02 ± 0.52 | 85.97 ± 0.47 |
| Blockwise | 86.82 ± 0.50 | 86.84 ± 0.45 |
| Spiral | 86.30 ± 0.31 | 86.29 ± 0.42 |

Snake의 평균 검증 accuracy는 Raster보다 약 0.23%p 높았다. Snake와 Spiral은 모든 좌표 이동 거리가 1픽셀이지만 평균 accuracy는 약 1.18%p 차이가 났다. Random은 평균 좌표 이동과 픽셀값 절대차가 가장 크고 검증 평균 성능이 가장 낮았다. 이 다섯 조건의 결과만으로 개별 신호 특성의 인과 효과를 분리할 수는 없다.

신호 지표는 공식 학습 이미지 전체 60,000장으로 계산했다. [신호·검증 지표 비교 CSV](signal_and_validation.csv)는 조건별로 이 지표를 검증 성능과 나란히 놓는다. [PSD와 경로 분석](../train-raster/)에서 주파수 전체와 lag별 반복 지표를 확인할 수 있다. [Seed별 결과 JSON](combined_validation.json)에는 각 seed의 선택 epoch, 지표, 혼동행렬, 체크포인트 경로가 있다.

Raster seed 0·1은 daissv2의 RTX 4090에서 실행했고, 나머지 13개는 daissv3의 RTX PRO 6000 Blackwell에서 실행했다. 두 서버의 PyTorch 버전도 달랐다. 체크포인트에 저장된 학습·검증 분할 인덱스는 15개 모두 동일했고, [체크포인트 매니페스트](../../checkpoints/manifest.json)에 실행 환경과 SHA-256을 기록했다. 공식 테스트 평가, 1D CNN baseline, 통제 실험은 이 결과에 포함되지 않는다.

## 검증 체크포인트 불러오기

공식 Fashion-MNIST IDX 파일을 [데이터 준비 안내](../../README.md)에 따라 받은 뒤, 저장소 루트에서 실행한다. `--data_root`는 체크포인트에 기록된 학습 서버의 경로를 현재 환경의 경로로 바꾼다.

```bash
python mamba_starter/evaluate.py \
  --checkpoint phase2_t5/checkpoints/snake/seed_0/best.pt \
  --data_root phase2_t5/data/FashionMNIST/raw \
  --split validation \
  --device cuda \
  --out phase2_t5/outputs/snake-seed0-validation.json
```
