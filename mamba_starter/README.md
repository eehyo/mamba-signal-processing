# Project Mamba · Student Starter

XAI639 (2026 Fall) 학생용 starter, version **1.1-student**. 과제 원문을 최종 기준으로 사용하세요.

이 코드는 데이터 로딩·공통 모델·학습·평가·실험 hooks를 제공합니다. 팀은 이를 읽고 이해한 뒤 **가설과 반증 기준, 새로운 구성요소, 대조 실험, 결과 분석과 해석**을 직접 설계합니다. Starter를 실행한 것만으로 새 구성요소 구현 요건을 충족하지는 않습니다.

## 설치 및 실행 확인

검증 환경은 Python 3.12 / PyTorch 2.5.1 CPU입니다. 아래 명령은 `mamba_starter/` 폴더에서 실행하세요.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q
python -m examples.hooks
```

Windows activation은 `.venv\Scripts\activate`입니다. GPU에서는 해당 환경과 호환되는 PyTorch/CUDA 설치를 별도로 확인해야 합니다. GPU 및 전체 데이터 장시간 학습은 이 배포본의 검증 범위에 포함되지 않습니다. `mamba_ssm`과 torchvision은 필요하지 않습니다.

데이터 다운로드 없이 짧게 학습·저장·재평가를 확인할 수 있습니다.

```bash
python train.py --dataset synthetic --seeds 0 --epochs 2 --train_subset 60 --d_model 8 --d_state 4 --n_layers 1 --out runs/smoke
python evaluate.py --checkpoint runs/smoke/seed_0/best.pt --out runs/smoke/reloaded.json
```

이 합성 데이터 실행은 작동 확인용입니다. 재실행할 때는 새로운 `--out` 경로를 쓰세요.

## 공통 학습 경로

```bash
python train.py --dataset har --data_root data/har --download --model mamba --seeds 0 1 2 --out runs/har_mamba
python train.py --dataset har --data_root data/har --model mamba --lti --seeds 0 1 2 --out runs/har_lti
python train.py --dataset har --data_root data/har --model cnn --seeds 0 1 2 --out runs/har_cnn
python train_ema.py --data_root data/har --seeds 0 1 2 --out runs/har_ema
```

`--model lstm`도 지원합니다. 전체 설정은 `python train.py --help`, 평가 옵션은 `python evaluate.py --help`로 확인하세요. 기본값은 20 epochs / batch64 / d_model32 / d_state16 / 2 layers입니다. 최적 hyperparameter를 뜻하지 않습니다. 내부 scan이 Python loop이므로 길이는 대략 200 이내로 유지하세요.

- HAR는 561차원 engineered features가 아닌 `[window,128,9]` inertial signals를 읽습니다. 채널 순서는 `body_acc_{x,y,z}`, `body_gyro_{x,y,z}`, `total_acc_{x,y,z}`입니다. 라벨은 0..5입니다.
- 공식 train subjects 중 일부를 validation으로 분리합니다. 기본 `split_seed=42`는 model seed와 별개입니다. `--val_subjects`로 명시할 수도 있습니다. 모든 비교 조건에서 같은 split을 유지하세요.
- 정규화는 training subset에서만 fit하고 checkpoint에 저장합니다. `--train_subset`을 쓰면 고정된 stratified subset을 선택합니다. 부분 데이터 사용을 보고서에 기록하세요.
- validation macro-F1로 epoch를 선택합니다. `summary.json`에 accuracy, macro-F1, confusion matrix(행=정답, 열=예측), seed별 결과와 sample std를 기록합니다. 한 seed의 std는 `null`입니다.
- EMA는 각 계수·채널의 mean/std/RMS를 feature로 사용합니다. 계수는 `--alphas`로 설정합니다. deterministic logistic regression은 seed별 값이 같을 수 있습니다.

## 파일과 API

| 파일 | 제공 기능 |
|---|---|
| `mamba_block.py` | selective/LTI SSM, Δ hooks, recording, `lti_kernel`, `causal_conv` |
| `models.py` | Mamba classifier, CNN, LSTM; padding을 제외한 pooling |
| `data.py` | HAR/Fashion loaders, subject split, normalization, irregular dataset/collate |
| `signal_utils.py` | `power_spectrum`, `ema`, `ema_features`, `downsample`, `upsample`, irregular utilities |
| `image_utils.py` | image resizing, raster/snake/column/random scans 및 row mode |
| `train.py`, `train_ema.py` | 공통 학습·validation·checkpoint |
| `evaluate.py` | PyTorch checkpoint 평가와 입력/Δ 개입 |
| `examples/hooks.py` | 기록과 불규칙 입력의 최소 사용 예제 |
| `tests/` | 수학·데이터·마스킹·재현성에 대한 구현 검사 |

```python
from models import MambaClassifier
model = MambaClassifier(input_dim=9, num_classes=6)
model.set_delta_scale(2.0)                  # 모든 layer의 절대 scale 값
model.set_record(True, states=True)
logits = model(x, dt_mult=gaps, mask=mask)  # x:[b,l,9], gaps/mask:[b,l]
record = model.records[0]                  # delta, A, B, C, mask, states
model.set_record(False)
```

record는 **최근 batch 하나만** CPU에 detach하여 저장합니다. 분석 시 valid mask를 적용하세요. Δ는 초가 아닌 내부 step이며, 기억 감쇠는 ΔA에 의존합니다.

`selective=False`/`--lti`는 **SSM 부분**의 Δ/B/C를 상수 parameter로 만듭니다. 전체 classifier의 SiLU/gating/RMSNorm은 유지됩니다. `block.ssm(u)`와 `causal_conv(u, block.lti_kernel(L))`를 비교할 수 있으며 u는 expanded-channel SSM 입력입니다. kernel은 lag0에 skip D를 포함합니다. 가변 `dt_mult`를 사용하면 고정 convolution kernel과의 등가성 조건이 성립하지 않습니다.

## 실험 hooks의 규칙

- **Rate 변화:** `evaluate.py --factor 2 --method aa` 등으로 원래 grid를 decimate합니다. HAR factor2는 25 Hz입니다. `--delta_scale`은 절대 scale이고, 생략하면 checkpoint 값을 유지합니다. 보정 조건과 control은 팀이 설계하세요.
- **Interpolation:** `--upsample linear` 또는 `poly`로 학습 grid에 되돌릴 수 있습니다. polyphase는 유한 FIR이며 경계 가정이 있습니다. aliasing 해석은 직접 수행하세요.
- **Irregular:** `--keep_rate`, `--gap_mode ignore/interpolate/delta/channel`. 첫/마지막 sample은 유지하며 첫 gap=1, 이후 gap=원래 index 차이입니다. 따라서 실현 keep rate는 설정값보다 조금 높습니다. 같은 missing seed로 전략 간 삭제 위치를 맞춥니다.
- **Gap channel:** 채널 수가 늘어나므로 해당 방식으로 별도 학습해야 합니다. CNN/LSTM은 Δ를 지원하지 않습니다. 관측 index 기준 convolution이 elapsed-time injection으로 자동 보정되는 것은 아닙니다.
- **Padding:** nonempty valid prefix 뒤에 오른쪽 padding만 허용합니다. 중간 결측은 관측값을 압축하고 gaps로 표현하세요. `valid_length_*`는 실제 관측 수입니다. irregular 입력의 `sample_rate_hz`는 null이며 기준 grid는 `reference_grid_rate_hz`에 기록됩니다.
- **Spectrum/EMA:** 함수 docstring에 축, 초기 상태, windowing, normalization을 정의했습니다. 클래스/채널별 집계, 그림, 해석 및 추가 분석 코드는 팀이 작성하세요.

## Fashion-MNIST

```bash
python train.py --dataset fashion --data_root data/fashion --download --image_size 14 --scan raster --train_subset 5000 --out runs/fashion
```

기본 pixel mode는 14×14=196 tokens, input_dim=1입니다. row mode는 행 하나가 token이고 width가 input_dim이어서 다른 width로 바로 평가할 수 없습니다. random permutation은 같은 해상도에서 모든 split에 고정됩니다.

기본 blur는 Gaussian σ=factor/2, reflect boundary, 4σ truncation 후 decimation입니다. `average`, `naive`도 제공합니다. 평가의 `--image_size 7 --image_filter ...`는 학습 시의 28→14 처리를 유지하고 그 결과에서 14→7로 변환합니다. 이 전처리 정의를 보고서에 기록하세요. 이미지 scan 주파수는 물리적 시간 주파수와 구분해야 합니다.

## 공식 test와 학생 구현 범위

개발·모델 선택·가설 수정에는 validation을 사용합니다. 프로토콜 확정 후에만 test를 평가하세요.

```bash
python evaluate.py --checkpoint runs/har_mamba/seed_0/best.pt --split test --final_protocol final_protocol.md --out runs/final_seed0.json
```

`final_protocol.md`에는 팀이 사전에 정한 conditions, seeds, checkpoints, metrics와 집계 규칙을 작성합니다. 도구는 문서의 hash를 기록하며 연구 절차 준수 자체를 보증하지 않습니다. `evaluate.py`는 PyTorch 전용입니다. EMA는 저장된 joblib의 `pipeline`에 동일한 `alphas`로 만든 `ema_features`를 넣어 별도 평가할 수 있습니다.

가설·반증 기준, 새로운 모델/분석 구성요소, 대조 실험, 세 seed 이상의 조건 비교, 그림과 해석은 팀이 완성해야 합니다. Conditional kernel, spectral occlusion, complex state, 새 scan/bidirectional 설계, CWRU loader 등은 제공하지 않습니다. 자세한 최소 수행 범위는 과제 원문을 확인하세요.

원본 mamba-minimal snapshot과 license는 `third_party/`, 수정/데이터 출처는 `NOTICE`에 있습니다.
