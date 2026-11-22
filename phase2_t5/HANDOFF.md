# Phase 2 T5 주 결과 및 인계

연구 질문은 스캔 순서가 픽셀 시퀀스의 주기성을 어떻게 바꾸며, 그 차이가 Mamba 분류 성능과 어떻게 연결되는가이다.

## 주 결과

Fashion-MNIST 학습 이미지 60,000장에 동일한 14×14 anti-aliased 전처리를 적용했다. 축 정렬 조건에는 Raster를 사용했다.

| 조건 | 평균 좌표 거리 (픽셀) | 평균 절대 픽셀값 차이 | 평균 픽셀 PSD 최대 비영 bin | 대응 주기 (인덱스) |
| --- | ---: | ---: | ---: | ---: |
| Axis-aligned (Raster) | 1.803 | 0.0959 | 14 | 14 |
| Snake | 1.000 | 0.0941 | 14 | 14 |
| Random | 7.172 | 0.2879 | 14 | 14 |
| Blockwise | 1.564 | 0.1088 | 7 | 28 |
| Spiral | 1.000 | 0.0720 | 1 | 196 |

Random의 14 인덱스는 고정 순열에서 최대였던 bin의 역수다. 해당 PSD 값은 약 0.451로 Raster의 약 7.611보다 낮으며, 최대 bin만으로 뚜렷한 반복을 판단할 수 없다. Spiral의 첫 비영 bin 역시 시퀀스 전체에 걸친 완만한 변화를 포함할 수 있다. 경로 주기성은 [이동 성분 PSD](results/train-raster/path_step_psd.csv)와 [lag별 이동 벡터 일치율](results/train-raster/path_periodicity.csv)에서 확인한다.

[경로](results/train-raster/paths.png) · [불연속성](results/train-raster/discontinuity.png) · [PSD](results/train-raster/psd.png) · [경로 주기성](results/train-raster/path_periodicity.png) · [순열](results/train-raster/scan_indices.npz)

[분류 실험 및 최종 평가 절차](PROTOCOL.md)

## 팀 역할

| 담당 | 범위 |
| --- | --- |
| 예림 | 다섯 스캔 변환, PSD·공간 불연속성·경로 주기성, 시각화 및 모델 입력 순열 인계 |
| 재효 | 다섯 스캔 × 3 seeds Mamba 분류, 1D CNN baseline, accuracy·macro-F1·혼동행렬 |
| 효정 | 통제 실험 및 causal convolution ablation, 성능 차이 해석 |

팀 최종 결과에는 최소 3개 모델 학습 seed의 평균과 표준편차, accuracy, macro-F1, 혼동행렬, baseline 비교를 포함한다. Random 순열 seed 42는 모델 학습 seed와 별개로 고정한다. 모델 결과가 준비되면 다섯 조건을 기준으로 이 표와 연결한다. 변환·신호 분석과 Mamba 학습·검증 15회가 완료됐다. 검증 지표 연결표, seed별 혼동행렬, 체크포인트 사용법은 [Mamba 검증 결과](results/mamba-validation/README.md)에 있다. 공식 테스트 결과와의 연결은 최종 평가 이후 진행한다.

Proposal에서 가설·반증 기준·실험 설계·역할을 최종 모델 평가 전에 확정한다. 예를 들어 스캔별 성능 차이가 공통 학습 seed에서 일관되는지, 신호 지표 차이와 같은 방향인지 사전에 정의한다. 성능이 포화해 seed 변동과 구분하기 어렵다면 모든 스캔에 동일한 난이도 조정을 검토한다. 최종 보고서와 발표에는 Mamba 기본 원리 설명을 포함한다.
