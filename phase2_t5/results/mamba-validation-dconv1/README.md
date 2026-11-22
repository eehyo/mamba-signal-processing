# Mamba causal convolution ablation

Fashion-MNIST 검증 이미지 12,000장에서 각 seed의 최적 검증 macro-F1 체크포인트를 평가했다. 표는 seed 0·1·2의 평균과 표본 표준편차이며 차이는 반올림 전 평균으로 계산했다.

| 스캔 | d_conv=1 accuracy (%) | d_conv=1 macro-F1 (%) | d_conv=4 macro-F1 (%) | 4→1 변화 (%p) |
|---|---:|---:|---:|---:|
| Raster | 85.71 ± 0.65 | 85.62 ± 0.72 | 87.22 | -1.59 |
| Snake | 85.94 ± 0.61 | 85.87 ± 0.62 | 87.50 | -1.63 |
| Random | 81.75 ± 0.87 | 81.58 ± 0.89 | 85.97 | -4.39 |
| Blockwise | 84.98 ± 0.43 | 85.04 ± 0.32 | 86.84 | -1.80 |
| Spiral | 84.79 ± 0.16 | 84.67 ± 0.19 | 86.29 | -1.62 |

커널을 4에서 1로 줄이면 Random의 평균 macro-F1이 가장 크게 하락했다. 네 구조적 스캔 각각에서 Random 대비 격차가 세 seed 모두 커졌다. Snake와 Random의 평균 격차는 1.53%p에서 4.29%p로 증가했다.

[seed별 지표와 혼동행렬](combined_validation.json) · [기본 모델 결과](../mamba-validation/README.md) · [체크포인트 SHA-256 및 환경](../../checkpoints-dconv1/manifest.json)

## 재현

GPU용 PyTorch 환경에서 저장소 루트를 기준으로 실행한다. 아래 데이터 경로에는 공식 학습 이미지와 라벨 IDX gzip 파일이 있어야 한다.

```bash
FASHION_DATA_ROOT="$PWD/mamba_starter/data/fashion" bash phase2_t5/run_t5.sh 1
```

다섯 스캔에서 각 seed를 20 epoch씩 학습한다. 결과는 `phase2_t5/outputs/t5-mamba-validation-dconv1/`에 저장된다. 기본 모델은 같은 스크립트에 인자 `4`를 사용한다.
