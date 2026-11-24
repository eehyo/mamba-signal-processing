# Phase 2 T5 공식 테스트 평가 확정 프로토콜

작성일: 2026-11-23. 이 문서는 공식 Fashion-MNIST 테스트셋을 처음 읽기 전에 작성했다. `mamba_starter/evaluate.py --split test --final_protocol phase2_t5/FINAL_PROTOCOL.md`가 이 파일의 SHA-256을 각 결과에 기록한다. 테스트 결과를 본 뒤에는 이 문서, 아래 체크포인트, 학습·분석 설정을 바꾸지 않는다.

## 1. 데이터와 전처리

- 평가 데이터: 공식 Fashion-MNIST 테스트 파일 `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`의 10,000장 전체.
- 전처리: 학습·검증과 같은 `t5_bilinear` 14×14 anti-aliased 축소([PROTOCOL.md](PROTOCOL.md)). 각 체크포인트에 저장된 스캔 순열과, 그 체크포인트의 학습 split에서 추정한 스칼라 평균·표준편차를 그대로 적용한다. Random은 학습 때와 같은 PCG64 seed 42 순열 하나를 쓴다.
- 평가는 배치 64, 섞지 않음, 모델은 eval 모드다.

## 2. 평가 대상 (모두 이미 학습이 끝난 체크포인트)

각 체크포인트는 seed별 검증 macro-F1 최고 epoch에서 저장한 `best.pt`다. 검증 결과만으로 선택했고, 테스트 결과에 따라 다시 고르지 않는다.

| 구분 | 모델 | 조건 | seed | 체크포인트 |
|---|---|---|---|---|
| 주 결과 | Mamba, `d_conv=4`, 20 epoch | Raster, Snake, Random, Blockwise, Spiral | 0·1·2 | `checkpoints/` (해시: `checkpoints/manifest.json`) |
| 주 결과 | 1D CNN baseline, kernel 5 × 2층, 20 epoch | 위 다섯 조건 | 0·1·2 | `outputs/t5-cnn-validation/` (해시: 부록) |
| 통제 실험 | Mamba, `d_conv=1`, 20 epoch | 위 다섯 조건 | 0·1·2 | `checkpoints-dconv1/` (해시: `checkpoints-dconv1/manifest.json`) |
| 추가 실험 | Mamba `d_conv=4`, 1D CNN 20 epoch | Spiral inward (Spiral 인덱스의 정확한 역순) | 0·1·2 | `outputs/t5-mamba-validation-extra/`, `outputs/t5-cnn-validation/spiral_inward/` (해시: 부록) |
| 보조 | 1D CNN, 최대 60 epoch, 검증 macro-F1이 5 epoch 연속 개선 없으면 중단 | 위 여섯 조건 | 0·1·2 | `outputs/t5-cnn-validation-e60/` (해시: 부록) |

Mamba와 CNN의 주 비교는 같은 20 epoch 예산의 결과로 한다. 60 epoch CNN은 학습 길이에 대한 보조 확인으로만 보고한다.

Column 스캔은 현재 학습되지 않았다. 나중에 [PROTOCOL.md](PROTOCOL.md)와 같은 설정(Mamba `d_conv=4`와 1D CNN, seed 0·1·2, 20 epoch, 검증 macro-F1로 체크포인트 선택)으로 학습하면, 이 문서를 바꾸지 않고 같은 절차로 한 번 평가한다. 보고할 때는 다른 조건보다 나중에 추가했다는 사실을 함께 적는다.

## 3. 지표와 집계

- 실행별: accuracy, macro-F1, 혼동행렬(행 = 정답, 열 = 예측), 혼동행렬에서 계산한 클래스별 F1.
- 조건별: 세 seed의 평균과 표본 표준편차(ddof=1).
- 조건 비교: 같은 seed끼리 짝지은 차이 (조건 − Raster). 세 seed의 부호가 모두 같을 때만 "일관된 차이"라고 부르며, 통계적 유의성은 주장하지 않는다.
- Spiral 방향: seed별 (Inward − Outward) macro-F1.
- `d_conv` 통제 실험: 각 구조적 스캔(Raster, Snake, Blockwise, Spiral)과 Random의 격차를 `d_conv=4`와 `d_conv=1`에서 각각 계산해 비교한다.
- 반증 기준과 가설은 [RESULTS_EXPLAINED.md](RESULTS_EXPLAINED.md) 4장의 것을 그대로 적용한다.

## 4. 범위

- 테스트셋은 위 체크포인트의 성능 지표를 계산하는 데만 쓴다. Δ 분석, 커널 분석, 예측 불일치 분석, 그림 선택은 검증 데이터 기준을 유지한다.
- 검증 결과와 테스트 결과는 구분해서 함께 보고한다.

## 부록: 로컬 체크포인트 SHA-256

```
136cf89a5c331ea490e8fbe202965fee4ac4a3ac4e1bc819fe48dc6011135116  phase2_t5/outputs/t5-mamba-validation-extra/spiral_inward_seed0/seed_0/best.pt
561aebaae3b5747ab367ed034d46850aabd0634073c9801490a2cedd39e5b4bd  phase2_t5/outputs/t5-mamba-validation-extra/spiral_inward_seed1/seed_1/best.pt
7ebbf876a729f2814537e86aa97603ab05429c9019dae8c3277e4140d15fcdcf  phase2_t5/outputs/t5-mamba-validation-extra/spiral_inward_seed2/seed_2/best.pt
240941592e3b4b0f355bf7eb69fa663febfd5b9a167fcb7d8fa1de8ce0637c37  phase2_t5/outputs/t5-cnn-validation/blockwise/seed_0/best.pt
1d9dd3e1af124b17b36b329db8a80270cb14fa5f8c7276ddf96c17a980087812  phase2_t5/outputs/t5-cnn-validation/blockwise/seed_1/best.pt
f147d02e7948b810bfeb3ad763d34ae4221281f3905be28803eee026c1aa2ea6  phase2_t5/outputs/t5-cnn-validation/blockwise/seed_2/best.pt
8a2cc3ceec117d1897180ec61fb9a0e057e98f7076dadf53081f5ef18de736ea  phase2_t5/outputs/t5-cnn-validation/random/seed_0/best.pt
a6ce27753e9c9212833c2dd5b95e0f6183e863f21c3cc926f57658b2d08a036f  phase2_t5/outputs/t5-cnn-validation/random/seed_1/best.pt
921c144f9a359ac997ad32f88d98bc15d5c014c350417b5b5c1f6acb718730e6  phase2_t5/outputs/t5-cnn-validation/random/seed_2/best.pt
4bee7f681251a4c4552f8da1d82059bbae4517c647d3a2d4040c99b5f4269150  phase2_t5/outputs/t5-cnn-validation/raster/seed_0/best.pt
86390127e17595ad39ed44a783271133d9944a8504ef204209583db94e3a2c9b  phase2_t5/outputs/t5-cnn-validation/raster/seed_1/best.pt
e550d78693c72499bcad7c763dbcec2dcc2675fca9d09e32d1f63e0e5e2c4450  phase2_t5/outputs/t5-cnn-validation/raster/seed_2/best.pt
b7e2140331e6b06832abb5c7e34cf3691ee1559f4de5c5c7ce0ab18f178d5132  phase2_t5/outputs/t5-cnn-validation/snake/seed_0/best.pt
da04c9145fc0562a4a15d1aebeebdb3b110ac6d1ddb60aaf894f14bc11b9c1ea  phase2_t5/outputs/t5-cnn-validation/snake/seed_1/best.pt
3188f1fa94c9f642886b382ceca8817391f78de7834d2b81453c2fa8207962c6  phase2_t5/outputs/t5-cnn-validation/snake/seed_2/best.pt
1aca10c2d70a36cb7e3ccbd0267bdd6ce2576793bd2aa2bf55f9eb85dc8a1c49  phase2_t5/outputs/t5-cnn-validation/spiral_inward/seed_0/best.pt
c8693194a9a68836fcdb7513f4a5216fdc250c7b4470df1a8e75410f56aede8b  phase2_t5/outputs/t5-cnn-validation/spiral_inward/seed_1/best.pt
29635e04da36994b6cdde3b5a239c39eee7cc66af541ed10d3dd60620151fb40  phase2_t5/outputs/t5-cnn-validation/spiral_inward/seed_2/best.pt
043e8c0616e13208cc22f04793d06b669ee48d7ded98d2eced0496b5276fe3ac  phase2_t5/outputs/t5-cnn-validation/spiral/seed_0/best.pt
2382c6965541a147c17cf134b05b1d2057ebc1059ad8fdfdfe76fc153cf60c40  phase2_t5/outputs/t5-cnn-validation/spiral/seed_1/best.pt
d820646536b2a96b03b2bd793d8e06c55a1d589ed9f9fd131f3b1a229b407cc2  phase2_t5/outputs/t5-cnn-validation/spiral/seed_2/best.pt
5e6a31992cb6984cfe6195c8ad97a03993f5c2f992a22c33e343ad027d46c3c9  phase2_t5/outputs/t5-cnn-validation-e60/blockwise/seed_0/best.pt
fbcf0232565b4bcd49129620ff10e904c01021c3e0cf926ca1af77b6414fcb11  phase2_t5/outputs/t5-cnn-validation-e60/blockwise/seed_1/best.pt
582298835afe33eed8025ba74e20902545834446fe043863cd53cfcc73229796  phase2_t5/outputs/t5-cnn-validation-e60/blockwise/seed_2/best.pt
a62ab452b0cbcf6953edbc9a24882c9212fe380dd6d223e18593570b59e4fcd5  phase2_t5/outputs/t5-cnn-validation-e60/random/seed_0/best.pt
a4d664911d71db0802dd333d8cb60b017d64843e47d5f42fbcbe82bdb9ccb618  phase2_t5/outputs/t5-cnn-validation-e60/random/seed_1/best.pt
be751b7f675bd473879116b3930ef8d24f9866d5447b368f1caf441dc01d98b8  phase2_t5/outputs/t5-cnn-validation-e60/random/seed_2/best.pt
088e9345990fd0e951150808a416fa8f1009d612b2aabfa997acab7d011327c9  phase2_t5/outputs/t5-cnn-validation-e60/raster/seed_0/best.pt
3b0a9456f33cfbdb6bbad28686ea4a35131cff24ecaeb0f0a262315feec14cf0  phase2_t5/outputs/t5-cnn-validation-e60/raster/seed_1/best.pt
998436a139ff94f8780a6538db15ae3ea38ab08edd566b58afeea718010e4130  phase2_t5/outputs/t5-cnn-validation-e60/raster/seed_2/best.pt
b8f94d3bfc4bebdd83b0d0d3b63e8691845b8e16e6c6ce836e91e15045e4e3f8  phase2_t5/outputs/t5-cnn-validation-e60/snake/seed_0/best.pt
c36d355270fc128f202f274af07de3ed1dd7314a9d4778ba539ae4346dc16bd9  phase2_t5/outputs/t5-cnn-validation-e60/snake/seed_1/best.pt
d8853988fd2529c74c0590d17e6b363926eb0738998d645462fe2395e7eabd75  phase2_t5/outputs/t5-cnn-validation-e60/snake/seed_2/best.pt
ed9817a3945494ddc06a1a92003a02ec95222504e23349eb542a6a73d7e4a0a8  phase2_t5/outputs/t5-cnn-validation-e60/spiral_inward/seed_0/best.pt
7ead5a1b21aa6079e26eaf2381a8b2e9b5047087adc5e0e69697cad2cbf471f8  phase2_t5/outputs/t5-cnn-validation-e60/spiral_inward/seed_1/best.pt
65922d0283de4c10d560bbf3effd59a0d06b78c6ef12eb20d9e5661edba24690  phase2_t5/outputs/t5-cnn-validation-e60/spiral_inward/seed_2/best.pt
95e5eb758aa10ff340099de0918e88c2db66d94fa4181cd961a0f7b3704e9be1  phase2_t5/outputs/t5-cnn-validation-e60/spiral/seed_0/best.pt
28d085ca0910973f7241bd25ff698faa2978451bd09900e5cdac9f1b7e2e9e11  phase2_t5/outputs/t5-cnn-validation-e60/spiral/seed_1/best.pt
b3c77d19c4713cae597402d27e7dc7a5f40454141308744f6717cb312f8aedce  phase2_t5/outputs/t5-cnn-validation-e60/spiral/seed_2/best.pt
```
