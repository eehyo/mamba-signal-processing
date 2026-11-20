# Phase 2 T5 · Fashion-MNIST 스캔 분석

## 데이터와 실행

Python에 NumPy, SciPy, Matplotlib이 필요하다. 공식 Fashion-MNIST IDX 압축 파일 4개를 `phase2_t5/data/FashionMNIST/raw/`에 둔다. 분석기는 압축 IDX와 TorchVision이 풀어둔 동일 이름의 비압축 IDX를 모두 읽는다. [Zalando Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

```bash
# 저장소 루트에서 실행
mkdir -p phase2_t5/data/FashionMNIST/raw
for name in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -fsSL "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/${name}.gz" \
    -o "phase2_t5/data/FashionMNIST/raw/${name}.gz" || break
done
cd phase2_t5
python ../mamba_starter/analyze_t5.py --split train --direction raster --output results/train-raster
python ../mamba_starter/analyze_t5.py --split train --direction column --output results/train-column
python ../mamba_starter/analyze_t5.py --split test --direction column --output results/test-column
```

두 방향을 직접 비교할 때는 같은 학습 split으로 실행한 `train-raster`와 `train-column`을 사용한다. 실행당 비교 조건은 다섯 가지다. `--data`로 IDX 폴더를, `--seed`로 Random seed를 지정할 수 있다.

## 주 결과와 방향별 참고

아래 재실행 명령은 CSV와 그림을 다시 생성한다. 제출된 결과를 유지하려면 별도 출력 디렉터리를 지정한다.

- 주 비교는 `results/train-raster/`의 다섯 조건이다. Axis-aligned의 대표 경로는 Raster다. 모든 조건에 동일한 학습 이미지 60,000장과 동일한 전처리·PSD 설정을 적용했다.
- `results/train-column/`은 같은 학습 split에서 Axis-aligned 방향만 바꾼 참고 실행이다. `results/test-column/`은 이미 열어본 **탐색적 테스트 참고**다. 둘 다 주 결과 집계에서 제외한다. 최종 테스트 분석에는 학습 split에서 고정한 Raster 설정과 분석 방법을 적용한다.
- Column의 최대 PSD가 첫 비영 bin에 위치했다. 196 인덱스는 해당 bin의 역수이며, 긴 규모의 변화나 추세도 같은 성분에 기여할 수 있다.

[주 결과표 및 팀 인계](HANDOFF.md)

## 모델 코드에서 사용

`phase2_t5/`에서 `PYTHONPATH=../mamba_starter python`으로 아래 예시를 실행한다.

```python
import numpy as np
from t5_preprocess import downsample, read_images
from t5_scan import apply_scan, scan_indices
from pathlib import Path

raw = read_images(Path("data/FashionMNIST/raw"), "train")
images = downsample(raw[:32])  # uint8 (32, 28, 28) → float32 (32, 14, 14)
indices = scan_indices("axis_aligned", direction="raster")
# 특정 분석 실행과 동일한 순서가 필요하면 저장한 순열을 사용한다.
with np.load("results/train-raster/scan_indices.npz") as orders:
    indices = orders["axis_aligned"]
sequences = apply_scan(images, indices)  # (32, 196)
model_input = sequences[..., None]        # (32, 196, 1), 채널 축이 필요한 모델
```

`apply_scan`은 이미 14×14로 전처리된 NumPy 배열 또는 PyTorch 텐서에서 마지막 두 공간 축을 순열에 따라 펼친다. dtype과 앞쪽 축을 보존한다. 예를 들어 PyTorch 입력이 `(B, 1, 14, 14)`라면 결과 `(B, 1, 196)`를 `.transpose(1, 2)`로 `(B, 196, 1)`로 바꿀 수 있다. 라벨과 이미지 순서는 기존 로더에서 유지한다.

## 순서와 계산 정의

- 인덱스는 0부터 시작하며 `row * 14 + column`이다. 각 순열을 생성할 때 길이와 전체 인덱스의 중복·누락 여부를 검사한다.
- Axis-aligned는 기본 Raster(행 우선) 또는 Column(열 우선) 하나를 사용한다. 방향은 메타데이터와 그림·CSV에 기록한다. Snake는 첫 행을 왼쪽부터 방문하고 이어지는 행마다 방향을 반전한다.
- Random은 PCG64와 기본 seed 42로 만든 단일 순열을 split 전체에 적용한다. Blockwise는 2×2 블록과 블록 내부를 모두 행 우선으로 방문한다.
- Spiral은 짝수 크기에서 가운데 2×2의 왼쪽 위인 `(6, 6)`에서 시작한다. 오른쪽, 아래, 왼쪽, 위 순으로 돌며 두 번 회전 방향을 진행할 때마다 이동 길이를 1씩 늘린다.
- 원본 uint8 픽셀을 [0, 1]로 변환한 뒤 각 축에 삼각형 저역통과 필터 `[1, 3, 3, 1]/8`을 적용해 절반 크기로 샘플링한다. 경계에서는 유효 픽셀 가중치 합으로 나눈다. 출력 픽셀 중심은 원본 좌표 `2i + 0.5`에 대응한다.
- 공간 불연속성은 리사이즈된 2D 격자상의 유클리드 거리와 정규화된 픽셀값의 절대차로 각각 계산한다. 끝과 시작을 연결하지 않는다. 좌표 통계는 고유 경로의 195개 이동에 대해, 픽셀값 통계는 선택한 split의 모든 이미지에서 나온 이동을 합쳐 계산한다. 표준편차는 모집단 기준이다.
- 픽셀 PSD는 각 길이 196 시퀀스의 평균을 제거하고, 직사각형 창을 사용하는 단측 periodogram을 구해 이미지별 PSD를 산술 평균한다. 시퀀스마다 분산을 1로 맞추는 추가 정규화는 적용하지 않는다. 주파수 단위는 cycles/scan index이며 주기는 양의 주파수의 역수다. 상위 5개는 DC를 제외하고 평균 PSD가 큰 bin으로 선정한다.
- 경로 PSD는 픽셀값 없이 195개 행·열 이동 성분에 같은 periodogram을 각각 적용한다. 별도로 각 lag에서 두 이동 벡터의 행·열 성분이 모두 일치하는 쌍의 비율을 구한다. 높은 lag에서 비교 가능한 쌍 수가 적어지므로 CSV에 쌍 수를 포함했다. 고정 방향 이동이 대부분인 경로는 여러 lag에서 일치율이 높을 수 있으므로 스펙트럼과 함께 해석한다.

## 산출물

| 파일 | 내용 |
|---|---|
| `metadata.json`, `scan_indices.npz` | 데이터 split, 실제 방향, seed, 규칙과 재사용 가능한 순열 |
| `paths.png` | 스캔 경로와 방문 인덱스 |
| `discontinuity.csv`, `discontinuity.png` | 두 불연속성 지표의 요약 통계와 분포 |
| `pixel_psd.csv`, `top_pixel_frequencies.csv`, `psd.png` | 픽셀 시퀀스 PSD와 상위 주파수 bin, 경로 PSD 비교 그림 |
| `path_step_psd.csv`, `path_periodicity.csv`, `path_periodicity.png` | 좌표 이동 성분의 PSD와 lag별 이동 벡터 반복 |

## Mamba 검증 실행
[분류 실험 및 최종 평가 절차](PROTOCOL.md)

`t5_bilinear`는 분석 코드와 동일한 14×14 전처리 및 다섯 순열을 적용한다. 이후 starter의 Standardizer는 학습 split에서만 스칼라 평균과 표준편차를 적합한다.

`CUDA_VISIBLE_DEVICES=0 bash phase2_t5/run_t5.sh`는 지정한 GPU에서 Raster, Snake, Random, Blockwise, Spiral 순으로 각각 모델 seed 0·1·2, 20 epoch를 실행한다. 공식 학습 데이터에서 stratified split으로 학습 48,000장과 검증 12,000장을 고정한다. 결과와 조건별 로그는 `phase2_t5/outputs/t5-mamba-validation/`에 저장된다. 완료된 조건은 `summary.json`에 seed별 지표, 혼동행렬 및 평균·표본 표준편차를 기록한다.
