# LLM 가중치 및 활성화 분포 분석 도구

이 도구는 LLM(Large Language Model)의 가중치와 활성화(activation)에서 크기(magnitude) 분포를 시각화하고 분석하는 데 사용됩니다. 특히 LLM.int8(), AWQ, SmoothQuant와 같은 논문에서 언급된 채널 단위 분포 특성을 확인할 수 있습니다.

## 기능

- LLM 모델의 가중치 크기 분포 시각화 (선형 및 로그 스케일)
- 추론 과정에서 각 레이어의 입력 및 출력 활성화 크기 분포 시각화
- Q, K, V 연산, FFN 등 다양한 연산 레이어별 분석
- 채널별 크기 분포 시각화
- 채널 단위 이상치(outlier) 분석 및 통계 제공

## 설치 방법

필요한 라이브러리를 설치합니다:

```bash
pip install torch transformers matplotlib numpy seaborn
```

## 사용 방법

1. `outlier_check.py` 파일에서 분석하고 싶은 모델 이름을 확인합니다:

```python
# 분석할 모델 설정 (기본값: Llama-2-7b-hf)
model_name = "meta-llama/Llama-2-7b-hf"
```

2. 스크립트를 실행합니다:

```bash
python outlier_check.py
```

3. 결과는 `magnitude_visualizations` 폴더에 저장됩니다.

## 분석 결과 이해하기

### 크기(Magnitude) 분포 시각화

- 각 가중치 및 활성화 텐서의 절대값 분포를 확인할 수 있습니다.
- 선형 스케일과 로그 스케일 두 가지 방식으로 시각화됩니다.
- 99%, 99.9% 백분위수가 표시되어 큰 값들의 분포를 확인할 수 있습니다.
- 다이나믹 레인지(최대값/최소값)가 계산되어 표시됩니다.

### 채널별 분석

- 각 채널의 평균 절대값 크기를 분석합니다.
- 특정 채널 전체가 이상치(outlier)인지 확인합니다 (LLM.int8(), AWQ, SmoothQuant 논문 참고).
- 채널별 크기 분포를 시각화하고, 이상치 채널을 강조표시합니다.
- 상위 N개 이상치 채널의 크기를 비교합니다.

### 요약 보고서

- 각 컴포넌트별 이상치 채널 비율을 보여주는 요약 그래프
- 텍스트 파일 형태의 채널 분석 요약 통계

## 커스터마이징

### 다른 모델 분석하기

다른 LLM 모델을 분석하려면 `model_name`을 변경하세요. 예:

```python
model_name = "EleutherAI/gpt-j-6B"  # 다른 모델
```

### 입력 텍스트 변경하기

추론 과정에서 사용될 입력 텍스트를 변경할 수 있습니다:

```python
sample_text = "원하는 입력 텍스트를 여기에 입력하세요."
```

### 특정 레이어에 초점 맞추기

특정 레이어만 분석하려면 `register_hooks` 메서드에서 다음 부분을 수정하세요:

```python
# 특정 레이어에만 초점 맞추기
if any(key in name for key in ["q_proj"]):  # 여기서 원하는 레이어만 선택
    hook = module.register_forward_hook(hook_fn(name))
    self.hooks.append(hook)
```

## 예제 코드

```python
from outlier_check import LLMMagnitudeAnalyzer

# 분석기 생성
analyzer = LLMMagnitudeAnalyzer("meta-llama/Llama-2-7b-hf")

# 전체 분석 실행
analyzer.run_full_analysis("분석에 사용할 입력 텍스트")

# 또는 단계별 실행
analyzer.load_model()
analyzer.register_hooks()
analyzer.run_inference("입력 텍스트")
analyzer.analyze_and_visualize_weights()
analyzer.analyze_and_visualize_activations()
```

## 출력 결과 파일

- `weight_magnitude_*.png`: 각 가중치 행렬의 크기 분포
- `weight_*_channel_magnitudes.png`: 각 가중치 행렬의 채널별 크기 분포
- `weight_*_top_outlier_channels.png`: 상위 이상치 채널 정보
- `activation_magnitude_*.png`: 각 활성화 텐서의 크기 분포
- `activation_*_channel_magnitudes.png`: 각 활성화 텐서의 채널별 크기 분포
- `weight_magnitude_comparison_*.png`: 동일한 타입의 가중치 크기 분포 비교
- `activation_magnitude_comparison_*.png`: 동일한 타입의 활성화 크기 분포 비교
- `weights_channel_analysis_summary.png`: 가중치 채널 분석 요약 그래프
- `activations_channel_analysis_summary.png`: 활성화 채널 분석 요약 그래프
- `weights_channel_analysis_summary.txt`: 가중치 채널 분석 요약 통계
- `activations_channel_analysis_summary.txt`: 활성화 채널 분석 요약 통계

## 참고 사항

- 대형 모델은 많은 메모리를 필요로 합니다. 충분한 GPU 메모리가 있는지 확인하세요.
- 채널 단위 이상치 기준은 평균 + 3*표준편차로 설정되어 있으며, 필요에 따라 조정할 수 있습니다.
- 다양한 양자화 기법들(LLM.int8(), AWQ, SmoothQuant 등)은 채널 단위 이상치 분포에 기반한 최적화를 수행합니다. 



# LLM 가중치 및 활성화 분포 분석 결과물 설명

분석 도구를 실행하면 `magnitude_visualizations` 폴더에 다양한 파일이 생성됩니다. 각 파일의 종류와 의미는 다음과 같습니다:

## 1. 가중치 크기 분포 시각화
**파일명**: `weight_magnitude_[레이어명].png`

**내용**:
- 좌측: 선형 스케일의 가중치 절대값 분포 히스토그램
- 우측: 로그 스케일의 가중치 절대값 분포 히스토그램
- 빨간선: 99% 백분위수 (상위 1%의 시작점)
- 녹색선: 99.9% 백분위수 (상위 0.1%의 시작점)

**의미**:
- 가중치 분포가 어떤 모양인지 파악 가능 (정규분포인지, 치우친 분포인지)
- 로그 스케일은 다양한 크기 범위의 값을 한 그래프에서 확인 가능
- 큰 값(99.9% 이상)의 비율과 크기를 확인 가능
- "Dynamic Range" 값은 최대값/최소값 비율로, 값이 클수록 양자화가 어려움

## 2. 채널별 크기 분포 시각화
**파일명**: `weight_[레이어명]_channel_magnitudes.png`

**내용**:
- 상단: 각 채널의 평균 절대값 크기를 바 차트로 표시 (빨간색 = 이상치 채널)
- 하단: 채널의 평균 크기 분포 히스토그램

**의미**:
- 각 채널별 평균 크기를 비교할 수 있음
- 이상치 채널(빨간색)은 평균보다 비정상적으로 큰 값을 가진 채널
- LLM.int8(), AWQ, SmoothQuant 논문에서 언급한 "전체 채널이 outlier"인 경우 확인
- 이상치 비율이 높은 레이어는 양자화 시 정확도 손실이 클 수 있음

## 3. 상위 이상치 채널 상세 정보
**파일명**: `weight_[레이어명]_top_outlier_channels.png`

**내용**:
- 상위 N개 이상치 채널의 평균 크기 비교

**의미**:
- 가장 큰 이상치 채널들을 식별하여 특히 관심을 기울여야 할 채널 확인
- 이런 채널들은 양자화 시 특별 처리가 필요할 수 있음

## 4. 활성화 크기 분포 시각화
**파일명**: `activation_magnitude_[레이어명]_[input/output]_[번호].png`

**내용**:
- 가중치와 동일한 형식이지만, 모델 추론 과정에서 발생한 활성화 값의 분포

**의미**:
- 실제 추론 시 각 레이어에서 발생하는 입출력 값의 크기 분포 파악
- 활성화 값의 다이나믹 레인지가 클 경우 양자화 어려움
- 입력과 출력 활성화를 비교하여 레이어가 값을 어떻게 변환하는지 파악

## 5. 컴포넌트 타입별 비교 시각화
**파일명**: 
- `weight_magnitude_comparison_[타입].png`
- `weight_magnitude_comparison_log_[타입].png`
- `activation_magnitude_comparison_[타입].png`
- `activation_magnitude_comparison_log_[타입].png`

**내용**:
- 같은 타입(query, key, value, FFN 등)의 여러 레이어 비교

**의미**:
- 모델 내 다른 레이어의 같은 컴포넌트가 유사한 분포를 가지는지 확인
- 특정 레이어/컴포넌트가 다른 패턴을 보이는지 식별
- 로그 스케일 버전은 작은 값의 분포까지 확인 가능

## 6. 채널 분석 요약
**파일명**: 
- `weights_channel_analysis_summary.png`
- `activations_channel_analysis_summary.png`

**내용**:
- 좌측: 각 컴포넌트별 이상치 채널 비율(%) 바 차트
- 우측: 전체 채널 수 대비 이상치 채널 수 산점도

**의미**:
- 전체 모델에서 이상치 채널 비율이 높은 컴포넌트 식별
- "10% Line"은 채널의 10%가 이상치인 기준선
- 이 선 위에 있는 컴포넌트는 양자화 시 특별 처리 필요

## 7. 상세 통계 텍스트 파일
**파일명**: 
- `weights_channel_analysis_summary.txt`
- `activations_channel_analysis_summary.txt`

**내용**:
- 각 컴포넌트의 채널 수, 이상치 채널 수, 이상치 비율(%) 테이블

**의미**:
- 정확한 수치로 된 상세 통계 확인
- 프로그래밍으로 추가 분석할 때 사용 가능

## 양자화 관련 의미

이 분석 결과들은 LLM.int8(), AWQ, SmoothQuant와 같은 양자화 기법에 중요한 정보를 제공합니다:

1. **채널 단위 이상치**: 특정 채널 전체가 다른 채널보다 훨씬 큰 값을 가진다면, 이 채널은 양자화 시 특별 처리가 필요합니다.

2. **동적 범위(Dynamic Range)**: 최대값과 최소값의 비율이 크면 양자화 시 정밀도 손실이 클 수 있습니다.

3. **분포 형태**: 비정규분포, 극단값이 많은 분포는 양자화가 어렵습니다.

4. **레이어별 차이**: 컴포넌트 타입별 비교를 통해 특정 레이어가 다른 패턴을 보이는지 확인할 수 있습니다.

이 정보들은 양자화 전략을 수립하거나, 양자화 기법의 타당성을 검증하는 데 활용할 수 있습니다.
