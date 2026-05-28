# ⚡ Oreum
> **SMP 저가 × EV 배터리 방전 시점 교차 예측 파이프라인**  

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-337AB7?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-latest-02569B?style=for-the-badge)
![Optuna](https://img.shields.io/badge/Optuna-TPE_Sampler-3B82F6?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF4B6E?style=for-the-badge)

---
## 📌 프로젝트 개요

제주 전력 계통 내 신재생에너지 공급 과잉과 전기차(EV) 확산이라는 변화가 동시에 나타남에 따라, 이를 효과적으로 연계·활용하기 위한 필요성에서 본 프로젝트를 기획하였습니다. 
**"SMP가 저렴한 순간에 배터리가 방전 임박한 EV에게 충전을 권고한다"** 는 단일 목표 아래, 세 가지 예측 모델을 순차 파이프라인으로 연결한 통합 AI 예측 시스템입니다.

<img width="833" height="729" alt="이거" src="https://github.com/user-attachments/assets/ca9787dd-087b-4cb8-a516-8fabfbdd7ac8" />

※ 파이프라인 다이어그램 시각화에는 Claude 기반 이미지 생성 기능을 활용했습니다.

---

### 핵심 알림 조건

| 조건 | 정의 |
|------|------|
| **SMP 저가** | Model B 예측 기준 향후 24h 내 SMP ≤ 80 원/kWh인 시간대 |
| **EV 방전 임박** | Model C 예측 기준 10% SoC 도달까지 남은 시간 ≤ 60분 |
| **충전 권고 발생** | `방전 예상 시각 ∈ SMP 저가 구간` 두 조건 동시 충족 시 🚨 발생 |

### 최종 성능 요약

| 모델 | 지표 | 달성값 | 목표 기준 |
|------|------|--------|-----------|
| Model A (전력수요) | MAPE | **2.05%** | ≤ 5% ✅ |
| Model A | RMSE | 15.91 MW | — |
| Model B (SMP) | MAPE | **4.28%** | ≤ 10% ✅ |
| Model B | RMSE | 6.45 원/kWh | — |
| Model C (EV 방전) | MAPE | **3.25%** | ≤ 15% ✅ |
| Model C | R² | **0.9939** | — |
| Pipeline A→B | MAPE | 7.28% | 운영 성능 |

---

## 🏗️ 시스템 아키텍처

<img width="826" height="994" alt="이거거" src="https://github.com/user-attachments/assets/d0b63c7a-67ae-4003-ac37-12ef880bc322" />

※ 아키텍처 다이어그램 시각화에는 Claude 기반 이미지 생성 기능을 활용했습니다.

---

## 📦 데이터 출처 및 전처리

### 주요 데이터 출처 개요

본 프로젝트에서 사용한 데이터는 **공공 데이터**와 **생성 데이터** 두 종류로 구분됩니다.
> 수집한 모든 데이터는 제주 시범 사업이 시작된 `24.6.1일 이후 데이터만 활용했습니다.<br>

| 구분 | 출처 | 데이터명 | 활용 모델 |
|------|------|----------|-----------|
| 공공 | 기상청 ASOS (기상자료개방포털 / 공공데이터포털) | 지상(종관) 기상관측자료 — 기온·풍속·습도·일사량 | Model A |
| 공공 | 한국전력거래소 / 공공데이터포털 | 전력수요 및 전력수급실적 데이터 | Model A, B |
| 공공 | 전력통계정보시스템(EPSIS) / 한국전력거래소 | 제주 SMP 데이터 | Model B |
| 공공 | 제주 관광 빅데이터 플랫폼 | 관광객 입도 수 데이터 | Model A |
| 공공 | 한국석유공사 Opinet | 국제 유가 데이터 (WTI / Dubai) | Model B |
| 공공 | 한국가스공사 / 공공데이터포털 | LNG 수입가격·천연가스 수입현황 데이터 | Model B |
| 생성 | 미니 ML 모델 (OOF 예측) | 재생에너지 발전비중 예측 피처 | Model B |
| 생성 | 도메인 지식 + 논문 기반 시뮬레이션 | EV 운행패턴·차량상태 데이터 (전량) | Model C |

> 데이터에 관한 자세한 내용은 첨부된 데이터 파일에서 확인 가능합니다.

---

### Model A 전처리 — 전력수요 & 기상 (`model_a_pre.ipynb`)

**입력 원천**: ASOS 4개 관측소(제주·고산·성산·서귀포) 시간별 기상 데이터, 전력수요 Wide Format CSV, 관광객 입도수, 통계청 산업생산지수

**주요 전처리 내용**

- **인코딩 방어 로드**: 원천 파일의 한글 깨짐 방지를 위해 `cp949` → `euc-kr` 순으로 우회 로드하는 방어적 코드 구현
- **결측치 처리**: 강수량·일사량은 현상 미발생을 의미하므로 `fillna(0)`, 기온·습도·풍속 등 연속 변수는 `interpolate(method='linear')` 적용
- **관측소 통합**: 4개 관측소 데이터를 날짜·시간 기준으로 평균 병합하여 제주도 전체 대표 기상 지표 산출
- **Wide → Long 변환**: `pd.melt()`로 1시~24시 컬럼을 단일 시계열 컬럼으로 변환, 미세 결측 약 20행은 `ffill()` 적용
- **Cyclic 변환**: 23시↔0시 불연속 문제 해결을 위해 삼각함수 기반 주기 임베딩 적용
  ```python
  df['hour_sin']  = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
  df['hour_cos']  = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
  df['month_sin'] = np.sin(2 * np.pi * df['datetime'].dt.month / 12)
  df['month_cos'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)
  ```
- **평년 기온 편차 산출**: 2000~2026년 장기 기후 데이터를 기반으로 월·일별 평년 평균 기온 사전을 생성하고 실제 기온과의 편차(`기온편차(°C)`) 파생변수 생성
- **Rolling 피처**: 최근 7일(168시간) 이동평균 수요 `rolling(window=168)` 생성
- **외부 데이터 융합**: 일별 관광객 수를 24시간 축으로 확장하는 Cross Join 적용, 월별 산업생산지수를 Transpose 후 시간별 프레임에 병합

**최종 산출물**: `model_a_data.csv` (16,056행)

---

### Model B 전처리 — SMP 가격 & 마이너스 구간 (`model_b_pre.ipynb`)

**입력 원천**: 제주/육지 SMP, 기상 예보, 신재생 발전량, WTI/Dubai 유가, USD/KRW 환율, 특수일 정보, 전력수급실적

**기준 시계열**: 2024-06-01 00:00 ~ 2026-03-31 23:00 (총 **16,056시간**, 누락 타임스탬프 = 0, 중복 행 = 0)

**주요 전처리 내용**

- **NFC 유니코드 정규화**: Windows/Mac 간 한글 자모 NFD 분리로 인한 경로 인식 에러 방지를 위해 `unicodedata.normalize('NFC', path)` 파이프라인 전면 배치
- **Hour-Ending 타임스탬프 변환**: 전력시장 1~24시 표기에서 `24:00`을 익일 `00:00`으로 귀속하는 `make_timestamp_hour_ending` 로직 자체 구현
- **기호 정제**: 금액·지수 내 콤마(`,`) 및 특수 대시(`−`) 제거 후 `float` 변환
- **Lag 피처 생성**: SMP 자기상관성 반영을 위해 1h / 24h / 48h / 168h 이전 가격 파생변수 생성
- **신재생 OOF 예측 피처 (핵심)**: 미래 시점의 실제 발전량을 직접 사용하면 Data Leakage가 발생하므로, 기상 예보·달력 변수만을 입력으로 하는 보조 회귀 모델을 Walk-Forward OOF 방식(최소 60일 학습 윈도우)으로 학습하여 `renewable_forecast` 피처를 생성·바인딩
- **명절 Window 효과**: 설날·추석 등 민족 대명절 전후 3일간 변동 가중치 부여 (`build_special_day_features`)
- **경제 지표 보간**: 주말·휴일 공백 구간은 선형 보간으로 16,056시간 연속 축 완성

**최종 산출물**: `model_b_data.csv` (16,056행)

---

### Model C 전처리 — EV 배터리 방전 예측 (시뮬레이션 데이터 생성)

**데이터 성격**: 실차 배터리 로그 미확보로 인해 **도메인 지식과 7편의 학술 논문**에 기반한 20,000행 규모의 물리 시뮬레이션 합성 데이터를 전량 생성하였습니다.

#### 입력 변수 구성

| 변수명 | 범위 | 도메인 의미 |
|--------|------|-------------|
| `Current_SoC` | 20% ~ 95% | 현재 배터리 잔존 용량 |
| `Battery_SoH` | 85% ~ 100% | 배터리 노후도 (낮을수록 실효 용량 감소) |
| `Battery_Temp` | 15°C ~ 35°C | 배터리 팩 내부 작동 온도 |
| `Speed_Variance` | 5 ~ 40 | 급가속·급감속 등 공격적 운전 성향 지표 |
| `Route_Elevation` | 0m ~ 800m | 제주 한라산 간선도로 고지대 주행 반영 |
| `Ambient_Temp` | -5°C ~ 33°C | 제주 사계절 외부 기온 변동폭 |
| `Precipitation` | 0 ~ 20 mm | 강수에 따른 노면 마찰력 변화 |

#### 물리 기반 타겟 생성 공식

타겟 변수 `Target_Min_to_10` (10% SoC 도달까지 남은 분)은 단순 선형 수식이 아닌 열역학·역학 복합 공식으로 생성되었습니다.

```
Available_SoC    = Current_SoC - 10
HVAC Factor      : T < 5°C  → 히터 부하 계수 +1.5
                   T > 28°C → 에어컨 부하 계수 +0.8
Efficiency_Loss  = 1.0 + (15 - T_amb) × 0.02   # 저온 배터리 효율 저하
Elevation Factor = 1.0 + (Elevation / 1000) × 0.5
Driving Factor   = 1.0 + Speed_Variance / 100
BMS Noise        = np.random.uniform(0.95, 1.05)  # ±5% 센서 불확실성 주입
```

#### 학술 논문 근거 (데이터 생성 기반)

| 논문 | 핵심 적용 내용 |
|------|---------------|
| Kumaresan & Rammohan, *Heliyon* 10 (2024) | 온도별 배터리 효율 보정 계수 설계 — 5°C 미만 0.8, 35°C 초과 0.9 |
| Poal et al., SAE Technical Paper (2021) | 고온·저온 HVAC 시스템이 배터리 소모에 미치는 영향 정량화 |
| Mądziel et al., *Energies* 18 (2025) | 기상 조건·계절과 EV 에너지 소비 간 데이터 기반 모델링 |
| Tiwary et al., IEEE NPSC (2022) | 속도 분산·급가속 등 운전 행동 피처와 에너지 소비 관계 |
| Al-Wreikat et al., *Applied Energy* 297 (2021) | 운전 행동·도로 경사·교통 조건의 실측 에너지 소비 영향 |
| Rachavelpula & Cha, arXiv (2026) | 운전자 군집화 기반 개인화 에너지 소비 추정 프레임워크 |
| Severson et al., *Nature Energy* 4 (2019) | BMS 센서 ±5% 측정 불확실성 — 노이즈 주입 근거 |

> **데이터 한계 및 향후 계획**: Model C는 실차 OBD-II / BMS CAN 데이터 없이 합성 데이터로 학습된 PoC 수준의 결과입니다. 현대차그룹의 제주 V2G 시범서비스(아이오닉9·EV9 55대, 2024.12~) 실차 로그 확보 시 즉시 재학습·재검증이 필요합니다.

**최종 산출물**: `model_c_data.csv` (20,000행)

---

## ⚙️ 핵심 구현 및 최적화

### Model A / B — Temporal Fusion Transformer

TFT 아키텍처를 선택한 핵심 이유는 시계열의 **변수 중요도 해석 가능성(Attention)**과 **분위수 예측(Quantile Loss)** 지원입니다.

```python
TemporalFusionTransformer.from_dataset(
    ds,
    learning_rate       = 5e-4,
    hidden_size         = 64,
    attention_head_size = 4,
    dropout             = 0.1,   # Model B: 0.15 (SMP 변동성 반영)
    hidden_continuous_size = 32,
    output_size         = 7,     # 7분위수 예측
    loss                = QuantileLoss(),
    weight_decay        = 1e-4,
)
```

Model B는 **연쇄 예측 구조**로 Model A의 24h 전력수요 예측값을 `predicted_demand` 피처로 직접 주입합니다. 이를 통해 전력 시스템의 인과관계(수요 → 가격)를 모델 구조 레벨에서 반영하였습니다.

### Sliding Window Cross-Validation

시계열 데이터 특성상 미래 데이터 참조를 원천 차단하기 위해 표준 K-Fold 대신 **Sliding Window CV**를 자체 구현하였습니다. 검증 구간이 항상 훈련 구간 이후에 위치하도록 설계하여 실제 운영 환경과 동일한 조건에서 성능을 검증합니다. Model A는 여름철 냉방 피크 과소학습 방지를 위해 6·7·8월 샘플에 3배 가중치를 부여하였습니다.

```python
def create_sliding_window_folds(df, time_col, train_size, gap, val_size, step):
    # train_size = 24 * 90  (90일)
    # gap        = 24       (예측 선행 시간)
    # val_size   = 24 * 14  (14일)
    # step       = 1000
```

### Model C — XGBoost + LightGBM Stacking (Optuna HPO)

물리 시뮬레이션 기반 합성 데이터 20,000행에 대해 두 트리 앙상블 모델의 예측을 Ridge 메타 러너로 결합하는 **2단 스태킹** 구조를 채택하였습니다.

```python
# Optuna TPE 탐색 공간 (XGBoost 기준)
params = {
    'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
    'max_depth':        trial.suggest_int('max_depth', 3, 9),
    'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
}
# 100 trials, early stopping 50 rounds
```

### Walk-Forward OOF — Data Leakage 원천 차단

Model B에서 신재생 발전량 실적을 직접 피처로 사용하면 **미래 정보 누출(Data Leakage)**이 발생합니다. 이를 방지하기 위해 기상 예보·달력 변수만으로 신재생 발전량을 예측하는 보조 모델을 별도 구축하고, **Walk-Forward OOF** 방식(최소 60일 학습 윈도우)으로 생성한 가상 예측치 `renewable_forecast`를 피처로 바인딩하였습니다.

---

## 🛠️ 기술 스택

### 딥러닝

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch_Lightning-792EE5?style=flat&logo=lightning&logoColor=white)
![TFT](https://img.shields.io/badge/pytorch--forecasting_(TFT)-EE4C2C?style=flat&logo=pytorch&logoColor=white)

### 트리 앙상블

![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat&logo=lightgbm&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn_(StackingRegressor)-F7931E?style=flat&logo=scikit-learn&logoColor=white)

### 하이퍼파라미터 최적화

![Optuna](https://img.shields.io/badge/Optuna_(TPE_Sampler)-4A90D9?style=flat&logo=optuna&logoColor=white)

### 모델 해석

![SHAP](https://img.shields.io/badge/SHAP-FF6B6B?style=flat&logoColor=white)
![UMAP](https://img.shields.io/badge/UMAP-6C5CE7?style=flat&logoColor=white)

### 데이터 처리

![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

### 시각화

![matplotlib](https://img.shields.io/badge/matplotlib-11557C?style=flat&logo=python&logoColor=white)
![seaborn](https://img.shields.io/badge/seaborn-4C8CBF?style=flat&logo=python&logoColor=white)

### 학습 인프라

![Google Colab](https://img.shields.io/badge/Google_Colab_(NVIDIA_Tesla_T4,_CUDA_12.8)-F9AB00?style=flat&logo=googlecolab&logoColor=white)

### 언어

![Python](https://img.shields.io/badge/Python_3.12.13-3776AB?style=flat&logo=python&logoColor=white)

---

## 🚀 설치 및 실행 방법

### 1. 환경 요구사항

- Python 3.10 이상
- CUDA 지원 GPU 권장 (Model A, B TFT 학습)
- Google Drive 연동 (데이터 경로 기반)

### 2. 패키지 설치

```bash
# Colab 환경
!pip install -q pytorch-forecasting pytorch-lightning
!pip install -q xgboost lightgbm shap optuna umap-learn scikit-learn
!apt-get install -y fonts-nanum -q

# 로컬 환경
pip install pytorch-forecasting pytorch-lightning xgboost lightgbm \
            shap optuna umap-learn scikit-learn matplotlib seaborn
```

### 3. 데이터 경로 설정

`Config` 클래스에서 Google Drive 마운트 경로를 수정합니다.

```python
class Config:
    BASE_PATH = '/content/drive/MyDrive/ax_team/'
    PATH_A    = BASE_PATH + 'ax_data/model_a/model_a_data.csv'
    PATH_B    = BASE_PATH + 'ax_data/model_b/model_b_data.csv'
    PATH_C    = BASE_PATH + 'ax_data/model_c/model_c_data.csv'
    SAVE_PATH = BASE_PATH + 'ax_modeling/'
```

### 4. 실행 순서

```
0. 환경 설정 (패키지 설치, 라이브러리 임포트)
1. 데이터 로드 및 전처리 (Model A / B / C)
2. EDA 시각화
3. TFT 공통 설정 (피처 정의, DataLoader 구성)
4. Sliding Window CV + 전체 학습 (Model A / B)
5. Model C 학습 — Optuna HPO → XGB + LGB Stacking
6. 모델 평가 (MAPE / RMSE / MAE / R²)
7. 핵심 시나리오 분석 (SMP 저가 × EV 방전 교차 감지)
8. SHAP / UMAP 시각화
9. 플릿 알림 시뮬레이션 (EV-001 ~ EV-004)
10. Optuna 수렴 곡선 & CV 학습 곡선 시각화
11. 최종 요약 & 모델 저장
```

---

## 📁 폴더 구조

```
/content/drive/MyDrive/ax_team/
│
├── ax_prepro/
│   ├── model_a_pre.ipynb
│   ├── model_b_pre.ipynb
│   └── model_c_pre_gen.ipynb
│
├── ax_data/
│   ├── model_a/
│   │   └── model_a_data.csv              # 전력수요 + 기상 통합 데이터마트 (16,056행)
│   ├── model_b/
│   │   └── model_b_data.csv              # SMP + 신재생 + 거시경제 (16,056행)
│   └── model_c/
│       └── model_c_data.csv              # EV 물리 시뮬레이션 합성 데이터 (20,000행)
│
└── ax_modeling/
    ├── integrated_modeling.ipynb         # 통합 학습 파이프라인 노트북
    ├── tft_a_best.ckpt                   # Model A 최적 체크포인트
    ├── tft_b_best.ckpt                   # Model B 최적 체크포인트
    ├── stacking_c.pkl                    # Model C Stacking 모델
    └── scaler_c.pkl                      # Model C RobustScaler
```

---

## 🔧 주요 트러블슈팅

### 1. 실측 EV 데이터 부재 — 7편의 논문 기반 물리 시뮬레이션으로 대체

**문제:** 제주 전기차 실차의 OBD-II / BMS CAN 로그는 제조사 보안 정책 및 개인정보 이슈로 수집이 불가능했습니다. 초기에 단순 균등 난수(uniform random)로 타겟을 생성하는 방안을 시도하였으나, 온도·고도·운전 성향 간의 물리적 상호작용이 전혀 반영되지 않아 모델이 의미 있는 패턴을 학습할 수 없었습니다.

**해결:** 7편의 학술 논문에서 추출한 정량 계수를 기반으로 열역학·역학 복합 공식을 직접 설계하여 20,000행의 물리 시뮬레이션 합성 데이터를 전량 생성하였습니다.

```python
def calculate_target_perfect(row):
    usable_soc = row['Current_SoC'] - 10

    hvac_load = 0
    if row['Ambient_Temp'] < 5:    hvac_load = 1.5
    elif row['Ambient_Temp'] > 28: hvac_load = 0.8

    rain_factor    = 1.05 if row['Precipitation'] > 5 else 1.0
    traffic_factor = 1.0 + (row['Traffic_Index'] * 0.02)
    temp_factor    = 1.0 if row['Ambient_Temp'] > 15 \
                     else 1.0 + (15 - row['Ambient_Temp']) * 0.02
    elev_factor    = 1.0 + (row['Route_Elevation'] / 1000) * 0.5
    speed_factor   = 1.0 + (row['Speed_Variance'] / 100)

    consumption_rate = (10 / row['Avg_Energy_Cons']) \
        * temp_factor * elev_factor * speed_factor \
        * rain_factor * traffic_factor

    target_min = (usable_soc * 5) / (consumption_rate + hvac_load)

    noise = np.random.uniform(0.95, 1.05)
    return round(max(target_min * noise, 5), 1)
```

**정성적 평가:**
단순 균등 난수 기반 타겟을 사용했을 때는 피처와 타겟 간 상관관계가 사실상 없어 스태킹 모델의 R²가 0.35 수준에 머물렀고, MAPE도 38%대로 정확한 예측이 불가한 수준이었습니다. 물리 공식 기반 합성으로 전환한 이후 R²=0.9939, MAPE=3.25%로 개선되었습니다. 특히 저온 구간(-5°C)에서 잔존 주행 시간이 단축되고, 고도 높은 경로에서 소비량이 증가하는 패턴이 도메인 기대치와 일치하는 방향으로 학습되었다는 점에서 단순 성능 수치 이상의 타당성을 확인할 수 있었습니다. BMS 센서 노이즈(±5%) 주입은 과적합을 억제하는 정규화 효과도 부수적으로 제공하였습니다.

다만 논문 계수가 특정 차종·기후 조건에서 도출된 값임을 감안할 때, 제주 현지 실차 데이터와의 계수 정합성 검증은 여전히 필요한 과제로 남습니다.

**한계:** 현대차그룹 제주 V2G 시범서비스(아이오닉9·EV9 55대, 2024.12~) 실차 로그 확보 시 즉시 재학습·재검증이 필요합니다.

---

### 2. Model B 피처 전처리 — 파생 피처의 inf / NaN 오염

**문제:** `solar_wind_ratio`, `net_load`, `renewable_pen` 등 나눗셈 기반 파생 피처는 분모가 0에 가까워지거나 SMP 음수 구간 진입 시 `inf` 및 `NaN`을 발생시켰습니다. 초기에는 단순 `fillna(0)` 처리만 적용하였는데, TFT 입력까지 오염이 전파되어 학습 2~3 에포크 이내에 손실값이 `nan`으로 폭발하는 현상이 반복되었고, 매번 학습을 강제 종료하고 재시작해야 했습니다.

**해결:** 분모에 `1e-8` epsilon을 더해 0 나눗셈을 원천 방지하고, 핵심 3개 컬럼에 대해 `inf → NaN 치환 → 선형 보간 → bfill/ffill → fillna(0)` 4단계 클리닝 파이프라인을 명시적으로 적용하였습니다.

```python
df_b['solar_wind_ratio'] = (df_b['forecast_solar'] /
                             (df_b['forecast_wind'] + 1e-8)).clip(upper=100)

for _col in ['predicted_demand', 'net_load', 'renewable_pen']:
    if _col in df_b.columns:
        df_b[_col] = pd.to_numeric(df_b[_col], errors='coerce')
        df_b[_col] = df_b[_col].replace([np.inf, -np.inf], np.nan)
        df_b[_col] = df_b[_col].interpolate(method='linear').bfill().ffill().fillna(0)
```

Sliding Window CV 생성 단계에서도 각 fold의 `train_df` / `val_df`에 동일한 보간 로직을 적용하여 구간 분할 후 생기는 경계 결측을 방지하였습니다.

```python
train_df = train_df.interpolate(method='linear', limit_direction='forward').bfill()
val_df   = val_df.interpolate(method='linear', limit_direction='forward').bfill()
```

**결과:**
`fillna(0)` 단일 처리 시에는 SMP 음수 구간(신재생 공급 과잉 시간대)에서 `renewable_pen` 이 비정상적으로 0으로 채워져 모델이 해당 구간의 패턴을 학습하지 못하고, 학습 손실이 에포크 2~3 부근에서 `nan`으로 발산하는 현상이 반복되었습니다. 4단계 클리닝 파이프라인 적용 이후에는 동일 구간에서 손실이 안정적으로 수렴하였고, 선형 보간 덕분에 인접 시간대의 맥락을 유지한 채 결측이 복원되어 SMP 음수 예측 정확도도 함께 개선되었습니다. CV fold 경계에도 동일 로직을 적용한 것이 특히 중요했는데, fold 분할 직후 발생하는 경계 샘플의 오염을 방치하면 특정 fold에서만 손실이 치솟는 불안정한 검증 결과가 나타날 수 있었기 때문입니다.

---

### 3. 여름철 냉방 피크 구간 검증 손실 과대 — WeightedRandomSampler 적용

**문제:** Sliding Window CV 분석 결과, 여름 구간(6~8월)을 검증 창으로 포함하는 fold의 `val_loss`가 비여름 fold 대비 약 8배 높게 나타났습니다. 전체 훈련 데이터 중 여름 샘플 비율이 약 25%에 불과해 연간 최대 수요를 형성하는 냉방 피크 패턴을 모델이 충분히 학습하지 못한 것이 원인이었습니다.

**해결:** PyTorch의 `WeightedRandomSampler`를 활용하여 6·7·8월 샘플 윈도우에 3배 가중치를 부여하였습니다.

```python
if 'Model_A' in model_name:
    sample_end_positions = tr_ds.index['index_end'].values
    sample_months = train_df['datetime'].iloc[sample_end_positions].dt.month.values

    weights = [3.0 if m in [6, 7, 8] else 1.0 for m in sample_months]

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    tr_dl = tr_ds.to_dataloader(train=True, batch_size=Config.BATCH_SIZE,
                                 num_workers=2, sampler=sampler, shuffle=False)
```

> `shuffle=False`는 WeightedRandomSampler와 DataLoader를 함께 사용할 때 샘플러 순서가 덮어써지는 것을 방지하기 위한 필수 설정입니다.

**정성적 평가:**
가중치 적용 전에는 여름 fold의 `val_loss`가 비여름 fold 대비 약 8배 높았고, 전체 MAPE도 11.08% 수준으로 목표치(5%)에 충족하지 않고 여름 피크 구간에서의 예측 오차가 운영 신뢰도를 떨어뜨리는 문제가 있었습니다. 3배 가중치 적용 이후 fold 간 `val_loss` 편차가 약 4.66배 수준으로 줄었고 전체 MAPE도 2.05%로 개선되었습니다. 다만 3배라는 배율은 도메인 직관에 기반한 휴리스틱 값으로, Optuna를 통해 가중치 배율 자체를 탐색 대상으로 삼으면 추가 개선 여지가 있습니다.

---

### 4. TFT 시계열 연속성 — 23시↔0시 단절

**문제:** 시간(0~23)을 정수 그대로 사용하면 23시와 0시 사이의 불연속이 발생합니다. 이로 인해 심야 수요 저점에서 새벽 수요 상승으로 전환되는 구간의 방향성을 모델이 반복적으로 역전 예측하는 현상이 관찰되었습니다.

**해결:** 사인·코사인 기반 Cyclic 변환으로 2차원 원형 임베딩을 적용하였습니다.

```python
df['hour_sin']  = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
df['hour_cos']  = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
df['month_sin'] = np.sin(2 * np.pi * df['datetime'].dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)
```

**정성적 평가:**
정수 인코딩 사용 시 모델이 23시에서 0시로의 전환을 "큰 값에서 작은 값으로의 급격한 변화"로 잘못 해석하여 자정 전후 수요 방향을 역전 예측하는 문제가 있었습니다. Cyclic 변환 적용 이후 이 경계 오차가 해소되었고, 이것이 Model A 전체 MAPE 2.05% 달성에 기여한 복합 요인 중 하나입니다. TFT 아키텍처 특성상 Attention 헤드가 시간 임베딩을 참조하여 패턴을 포착하는 구조이므로, 연속적인 원형 임베딩을 입력으로 제공하는 것이 정수 인코딩보다 구조적으로 더 적합한 선택이었습니다. 월별 임베딩(month_sin/cos) 역시 12월↔1월 계절 전환 경계에서 동일한 연속성 이점을 제공합니다.

---

## 📊 결과 해석

### 1. 그래프 해석

#### 1-1. EDA 대시보드

##### [Model B] 제주 SMP 추이 (최근 30일)
SMP는 대부분 100원/kWh 내외에서 등락하나, 특정 시점에 0 이하로 급락하는 음수 구간이 반복적으로 관측된다. 이는 재생에너지 출력 급증으로 공급이 수요를 초과할 때 발생하는 제주 계통 특유의 현상으로, 단순 시계열 모델이 아닌 계통 상황 변수를 결합한 예측이 필수적임을 시사한다. 저가 발생 시간대 분포를 보면 새벽 2~5시에 집중되어 있어, 심야 재생에너지 잉여 구간이 EV 저비용 충전의 핵심 타깃 시간대임을 확인할 수 있다.

##### [Model A] 전력수요 추이 (최근 30일)
700~900MW 범위에서 일중 두 개의 피크(오전·오후)를 갖는 뚜렷한 주기 패턴이 유지된다. 주말·공휴일에 진폭이 줄어드는 패턴도 확인되며, 이는 Decoder 변수 중요도에서 is_holiday가 SMP의 최강 예측 신호로 나타나는 결과와 일관된다.

##### [Model C] EV 10% SoC까지 남은 시간 분포
방전 잔여 시간은 우측으로 긴 꼬리를 가지는 비대칭 분포를 보인다. 최빈값은 20~40분 구간에 집중되어 있으며, 경보 임계값인 60분(붉은 점선)이 분포의 상위 약 40%에 위치한다. 전체 EV 상태 중 상당 비율이 실제로 경보 구간 이내에 있을 수 있음을 의미하며, 조기 경보 시스템의 실용 필요성을 수치로 뒷받침한다.

##### [Model C] 현재 SoC vs 방전 잔여 시간 (SoH 색상)
SoC와 잔여 방전 시간은 전반적으로 선형에 가까운 양의 관계를 보이나, 동일한 SoC 수준에서도 SoH에 따라 잔여 시간의 분산이 크게 벌어진다. SoH가 낮은 차량(붉은 계열)은 같은 SoC에서도 잔여 시간이 짧아지는 경향이 명확하며, 이는 SHAP 분석에서 soh_adjusted_capacity가 상위 5위에 포함되는 결과와 정합적이다. 동일 충전 상태라도 차량 노후도에 따라 실제 주행 가능 시간이 달라지는 효과를 모델이 포착하고 있음을 시각적으로 확인할 수 있다.

##### [Model B] SMP 피처 상관관계
jeju_smp와 smp_roll_mean24 간 양의 상관(0.56)이 가장 두드러진다. SMP가 강한 자기상관 구조를 가지며, 24시간 이동평균이 단기 예측의 기준선 역할을 한다는 의미다. 반면 공급예비율은 jeju_smp와 음의 상관(-0.34), smp_roll_mean24와도 음의 상관(-0.52)을 보여 예비력이 낮을수록 SMP가 상승하는 계통 경제학적 직관과 정확히 일치한다.

##### [Model C] XGBoost 피처 중요도 Top 10
Current_SoC가 중요도 0.45로 압도적 1위를 차지하며, soc_temp_interaction, Avg_Energy_Cons 순으로 이어진다. 상위 3개 변수가 전체 중요도의 과반을 설명하는 집중 구조이며, SoC 단독이 아닌 온도와의 상호작용 항이 독립적으로 높은 중요도를 갖는다는 점이 주목된다. "같은 충전 상태라도 기온이 낮으면 더 빨리 방전된다"는 물리적 현상을 모델이 명확히 학습했음을 의미한다.

--- 

#### 1-2. 통합 대시보드 

##### [Model C] 예측 vs 실제 Scatter (MAPE=3.2%, R²=0.994)
예측값과 실제값이 완벽 예측선(y=x) 주변에 매우 촘촘하게 분포하며, 전 범위에 걸쳐 이탈 없이 선형 관계를 유지한다. 특히 방전 잔여 시간이 짧은 구간(0~60분)에서도 산포가 크지 않아, 가장 위급한 상황에서의 예측 신뢰도가 충분히 확보되었음을 확인할 수 있다. 경보 임계인 60분 부근에서의 예측 정밀도는 알람 시스템의 오발령률과 직결되므로 이 구간의 밀집된 분포가 실용적으로 특히 중요하다.

##### [Model C] 잔차 플롯
잔차는 예측값 전 구간에 걸쳐 0 근방에 무작위하게 분포하며 뚜렷한 패턴이 없다. 다만 예측값이 커질수록(잔여 시간이 길수록) 잔차의 분산이 소폭 증가하는 이분산성 경향이 관찰된다. 이는 방전이 임박한 단거리 구간보다 장거리 잔여 시간 예측에서 불확실성이 상대적으로 높아지는 자연스러운 특성으로, 알람 발동이 목적인 단거리 구간에서는 실질적 영향이 제한적이다.

##### [Model C] LightGBM vs XGBoost 피처 중요도 비교
두 모델 모두 Current_SoC를 압도적 1위로 일치하여 선택하고 있으며, 상위 변수 구성도 대체로 일관된다. 다만 Ambient_Temp와 Route_Elevation의 순위가 모델 간에 소폭 교차하는 차이가 있다. 두 모델이 독립적으로 동일한 피처 중요도 구조에 수렴한다는 사실은 해당 변수들의 예측 기여가 모델 아키텍처에 의존하지 않는 실질적인 신호임을 강하게 시사한다.

##### 최종 MAPE 비교 Bar Chart
Model A(2.05%) → Model B(4.28%) → Model C(3.25%) → Pipeline A→B(7.28%) 순으로 오차가 증가하는 구조가 시각적으로 명확히 드러난다. 파이프라인 MAPE가 단독 Model B MAPE 대비 약 3%p 높은 것은 A의 예측 오차가 B의 입력으로 전파된 결과로, 파이프라인 전체 성능 관리에서 Model A의 정밀도 유지가 병목임을 보여준다.

##### [A/B] Sliding Window CV 손실
Model A(파란 선)는 Fold 7에서 Val Loss가 약 135로 급등하는 이상치를 보인다. 이 구간은 특정 계절 전환기 또는 이벤트 구간에 해당하는 시계열 분할일 가능성이 높으며, 해당 기간의 패턴이 다른 Fold와 구조적으로 이질적임을 의미한다. Model B(붉은 선)는 Fold 0~1에서 높은 손실을 보이다가 이후 안정화되는 패턴으로, 초기 학습 구간의 데이터 품질 또는 분포 이동이 원인으로 추정된다.

---

#### 1-3. SHAP 분석 — Model C EV 방전 시간 예측

##### SHAP Summary Plot 해석
Current_SoC는 SHAP 값이 -40 ~ +60 범위로 분포하며, 다른 모든 변수의 SHAP 범위가 ±10 내외인 것과 극명히 대비된다. 이 변수 하나가 예측 출력을 수십 분 단위로 좌우하는 지배적 신호임을 의미한다. 높은 SoC(붉은 점)는 오른쪽(잔여 시간 증가), 낮은 SoC(파란 점)는 왼쪽(방전 임박)으로 강하게 치우치는 단조로운 방향성을 보인다.

soc_temp_interaction은 높은 값에서 넓은 분산을 보여 비선형 상호작용 효과를 포착하고 있음이 분포 형태에서 드러난다. Ambient_Temp는 낮은 값(저온)에서 일관되게 음의 SHAP 기여를 보여 한파 환경에서 방전이 가속됨을 명확히 시각화하며, Route_Elevation은 값이 높을수록 음의 방향으로 기여하여 경로 특성이 에너지 소비량에 직접 반영됨을 확인할 수 있다.

##### Mean |SHAP| 순위

| 순위 | 변수 | 평균 \|SHAP\| | 물리적 의미 |
|---|---|---|---|
| 1 | Current_SoC | ~20 | 잔여 충전량이 방전 시간의 직접 결정 인자 |
| 2 | Avg_Energy_Cons | ~9 | 단위 시간당 소비 전력이 클수록 방전 가속 |
| 3 | soc_temp_interaction | ~8 | 저온에서 실효 용량 감소 — SoC만으로 설명 불가한 기온 효과 포착 |
| 4 | Ambient_Temp | ~7.5 | 외기온 직접 효과 (한파·폭염 모두 영향) |
| 5 | soh_adjusted_capacity | ~5 | 노후 배터리의 실질 가용 용량 감소 반영 |
| 6 | Route_Elevation | ~5 | 오르막 구간 비중이 높을수록 에너지 소비 급증 |
| 7 | traffic_energy_loss | ~3 | 정체 구간 공회전 손실 |
| 8 | Speed_Variance | ~2.5 | 급가속·급감속 패턴이 비효율 소비를 유발 |

---

#### 1-4. TFT Attention 및 변수 중요도 분석

##### Model A (전력수요) — Attention 패턴
Attention이 -175 ~ 0 전 구간에 걸쳐 비교적 균등하게 분포하되, -125 ~ -100 구간에서 피크가 집중된다. 모델이 약 4~5일 전 동시간대 패턴을 특히 중요한 참조 포인트로 활용함을 의미하며, 제주 전력수요의 주중 주기성이 모델 내부에 내재화된 결과로 해석된다.

##### Model A — Static 변수 중요도
group_id가 약 80%로 압도적 1위를 차지하고 encoder_length가 약 20%를 차지한다. 지역 그룹 식별자가 전력수요 예측의 기반 컨텍스트로 가장 강하게 작동하고 있으며, 이는 지역별 수요 구조 차이가 모델이 학습해야 할 가장 기본적인 정보임을 의미한다. 반대로 Model B에서는 encoder_length가 73%로 1위인데, 이는 SMP 예측이 지역 구분보다 과거 데이터의 양(= 얼마나 긴 과거를 참조하는가)에 더 강하게 의존하는 구조적 차이를 보여준다.

##### Model A — Encoder/Decoder 변수 중요도
Encoder 측에서 전기업 및 가스업 생산지수(~17%) 와 외국인방문객수(~13.5%) 가 압도적 1~2위를 차지한다. 일반적인 전력수요 모델에서 기온과 시간 변수가 최상위에 오르는 것과 달리, 제주 계통에서는 산업 생산 활동과 관광 수요가 기저 수요를 결정하는 주요 구동 인자임이 XAI를 통해 실증되었다. Decoder 측에서는 체감온도(~22%) 가 단연 1위로, 냉난방 수요가 미래 수요 변동의 핵심 경로임을 확인한다.

##### Model B (SMP) — Attention 패턴
Model A와 달리 -165 ~ -130 구간(약 5~7일 전)에서 attention이 높고, 최근 시점으로 갈수록 감소하는 역방향 패턴을 보인다. SMP는 연료 계약 주기·재생에너지 출력 예측 지평과 연동되어 과거 1주 전 계통 상황이 현재 SMP 형성에 강하게 영향을 미침을 모델이 학습한 결과다.

##### Model B — Static 변수 중요도
encoder_length가 약 73%로 1위를 차지한다. Model A의 group_id 중심 구조와 대조적으로, SMP 예측은 과거 시계열 데이터를 얼마나 길게 참조하느냐가 가장 결정적임을 의미한다. SMP가 연료비·환율·계통 이력의 누적 패턴을 반영하는 변수인 만큼, 긴 맥락 창이 필수적이라는 시장 구조와 일관된 결과다.

##### Model B — Encoder/Decoder 변수 중요도
Encoder에서 fuel_cost_unit(~11%), smp_roll_std24(~8.5%), net_load(~8%), usd_krw(~6.5%)가 상위를 점한다. 연료비·환율·계통 부하라는 SMP 결정 3요소가 고르게 반영된 것으로, 모델의 학습 내용이 전력시장 구조와 정합적이다. Decoder에서는 is_holiday(~21%) 가 압도적 1위로, 휴일 수요 감소 → SMP 하락 패턴이 미래 정보로서 가장 강력한 예측 신호임을 보여준다.

##### Model C — Optuna 수렴 곡선 및 CV Fold 손실
XGBoost(Best 3.1474)와 LightGBM(Best 3.1451) 모두 Trial 20~30 이후 Val RMSE가 약 3.15 수준으로 수렴하며 이후 거의 개선되지 않는다. 현재 피처셋과 모델 구조에서 달성 가능한 성능의 상한에 근접했음을 시사하며, 추가 성능 향상을 위해서는 하이퍼파라미터 탐색보다 피처 엔지니어링 또는 새로운 데이터 소스 도입이 더 효과적임을 의미한다.

Model A의 Fold 7에서 Val Loss가 약 135로 급등하는 이상치는 해당 시계열 구간이 다른 Fold와 구조적으로 이질적인 패턴(계절 전환기, 특수 이벤트 등)을 포함할 가능성을 시사한다. Model B는 Fold 0~1에서 손실이 높다가 이후 안정화되어, 초기 훈련 구간의 데이터 분포가 이후 구간과 다소 상이함을 나타낸다.

---

### 2. 시나리오 시뮬레이션 결과

#### 2-1. 시나리오별 방전 및 SMP 예측 비교

| 시나리오 | 조건 | 방전 잔여 시간 | SMP 저가 교차 | 알람 |
|---|---|---|---|---|
| 시나리오 1 — 기준 | 평상 조건 | 130분 | 없음 | ❌ |
| 시나리오 2 — 재생에너지↑ | 태양광·풍력 출력 증가 | 28분 | 없음 | 🚨 |
| 시나리오 3 — 폭염 | 고온·냉방 수요 급증 | 57분 | 없음 | 🚨 |

##### 시나리오 1 (기준)
방전 잔여 시간 130분으로 경보 임계(60분)에 충분한 여유가 있다. SMP도 안정적 구간을 유지해 특별한 조치가 불필요한 정상 운영 상태다.

##### 시나리오 2 (재생에너지↑)
태양광·풍력 출력 증가로 순부하가 급감하면서 SMP 하락 압력이 발생하는 동시에 방전 잔여 시간이 28분까지 단축된다. 전력이 가장 저렴한 바로 그 순간에 충전 필요성이 가장 급박해지는 구조로, 재생에너지 잉여 흡수와 방전 방지를 동시에 달성하는 최적 자동 충전 발동 조건이 시스템에 의해 감지된다.

##### 시나리오 3 (폭염)
체감온도 상승이 냉방 에너지 소비 증가와 배터리 열화를 동시에 유발하여 방전 잔여 시간이 57분으로 단축된다. SHAP 분석에서 Ambient_Temp와 soc_temp_interaction이 상위 변수로 확인된 것과 일관된 결과이며, 여름철 고온 환경이 EV 운용 리스크를 구체적인 수치로 현실화한 시나리오다.

---

#### 2-2. [B+C] 방전 예상 시각 × SMP 오버레이 분석

세 시나리오의 방전 예상 도달 시각을 SMP 24시간 예측 곡선과 중첩하면 시나리오별 충전 전략이 명확히 도출된다.

- 기준: 방전 도달 예상(~1.2시간 후) 시점에 SMP가 약 120원/kWh 이상 고가 구간. 여유 시간이 있으므로 SMP 하락을 기다리는 충전 지연 전략이 유효하다.
- 재생에너지↑: 방전 도달 예상(~0.5시간 후) 시점이 SMP 저가 구간과 근접. 비용 최적 충전 타이밍과 방전 위기 시점이 겹치는 즉각 충전 발동 조건이 성립한다.
- 폭염: 방전 도달 예상(~1.0시간 후) 시점에 SMP가 고가 구간 진입. 충전 비용 최고조 시점에 강제 충전이 필요해지는 최악의 조합으로, 사전 앞당기기 충전이 필요함을 시스템이 경고한다.

---

#### 2-3. 플릿 알람 현황 — 시나리오 4 (심야 재생에너지 과잉)

| EV ID | 방전 잔여 시간 | SMP 저가 교차 | 알람 | 권장 조치 |
|---|---|---|---|---|
| EV-001 | 79분 | ❌ | 없음 | 모니터링 유지, 다음 저가 구간 대기 |
| EV-002 | 28분 | ✅ | 🚨 | 저가 구간 즉시 충전 명령 |
| EV-003 | 103분 | ❌ | 없음 | 여유 충분 — 최적 타이밍 대기 |
| EV-004 | 19분 | ✅ | 🚨 | 최우선 긴급 충전 배차 |

EV-004는 방전 잔여 19분으로 플릿 내 가장 위급한 상태이며, SMP 저가 교차까지 성립해 즉각 충전이 비용과 운용 리스크를 동시에 해소한다. EV-002와 동시 충전 배차가 가능하다. 반면 EV-001·EV-003은 방전 여유가 60분을 상회하므로 다음 저가 구간까지 충전을 지연하는 비용 최적화 전략을 적용할 수 있다.

---

### 3. 시스템의 실질적 기여 가치

#### 3-1. 전력망 계통 안정화
재생에너지 잉여가 예상되는 시간대를 사전에 감지하여 EV 충전을 해당 구간으로 유도함으로써, EV를 잉여 전력 흡수용 유연성 자원으로 자동 활용하는 체계를 구현한다. TFT Decoder 분석에서 is_holiday와 temperature_forecast가 강력한 예측 신호로 확인됨에 따라, 공휴일 수요 급감이나 기상 이변 상황에서의 SMP 급변을 24~48시간 전에 계통 운영자에게 전달하는 사전 경보 인프라로 기능할 수 있다.

#### 3-2. EV 플릿 운영 비용 절감
차량별 방전 잔여 시간과 SMP 예측을 결합하여 방전 위험을 회피하면서 동시에 충전 비용이 가장 낮은 타이밍에 자동으로 충전 명령을 발행하는 최적화 스케줄링을 가능하게 한다. 시나리오 4에서 EV-003처럼 여유가 있는 차량은 충전을 지연하고, EV-004처럼 임박한 차량은 즉각 배차하는 차량 상태 기반 우선순위 자동 분류 기능이 플릿 규모가 클수록 누적 비용 절감 효과를 기하급수적으로 증폭시킨다.

#### 3-3. 배터리 열화 예측 및 선제 유지보수
SHAP 분석에서 soh_adjusted_capacity와 Battery_SoH가 방전 예측에 독립적인 기여를 보임은, 동일 SoC라도 차량 노후도에 따라 잔여 시간이 달라지는 효과를 모델이 실질적으로 포착하고 있음을 의미한다. 이를 바탕으로 SoH가 임계 이하인 차량에는 더 이른 시점에 경보를 발동하는 개인화된 경보 임계 조정이 가능하며, SoH 추이 누적 추적을 통해 배터리 교체 시점을 예측하는 예측적 유지보수(Predictive Maintenance) 체계로 확장할 수 있다.

#### 3-4. 수요반응(DR) 프로그램 정밀화
Model A Encoder에서 외국인방문객수가 13.5%로 2위를 차지한다는 발견은 제주 특화 수요 구조를 XAI로 설명하는 근거가 된다. 관광 성수기·비수기에 따른 기저 수요 변동을 모델이 포착하고 있으므로, 단순 기온 기반 DR 발동 기준을 넘어 관광·산업 활동 지표를 연동한 제주형 맞춤 수요반응 트리거 설계가 가능하다. 또한 폭염 시나리오에서 방전 잔여 시간이 130분→57분으로 단축되는 결과는, 기상 경보와 연동하여 EV 충전 우선순위를 자동 상향하는 기상 연동 DR 체계의 기술적 근거를 제공한다.

#### 3-5. 에너지 시장 V2G 수익화
A→B→C 파이프라인의 가장 직접적인 수익화 경로는 SMP 저가 매수 + 고가 방전(V2G) 차익이다. 시나리오 2에서 확인된 것처럼 재생에너지 잉여로 SMP가 낮아지는 구간에 충전하고, 피크 수요로 SMP가 높아지는 구간에 V2G로 역공급하면 충전-방전 가격 차를 수익으로 전환할 수 있다. 이를 위해서는 수 시간 전 SMP 방향성과 EV 배터리 가용 용량을 동시에 예측하는 통합 파이프라인이 전제 조건이 되며, 본 시스템은 그 핵심 인프라로 기능한다.

---

## ✅ 결과 및 그래프

<img width="1295" height="924" alt="image" src="https://github.com/user-attachments/assets/192cdb9a-e2cc-4c6a-a039-bec8d7100975" />

---

<img width="1204" height="1004" alt="image" src="https://github.com/user-attachments/assets/5f66775c-6ce9-4ab0-8385-848e24a58f8c" />

---

<img width="790" height="939" alt="image" src="https://github.com/user-attachments/assets/1e53c161-008b-484a-aa06-7447e6085524" />

---

<img width="1189" height="593" alt="image" src="https://github.com/user-attachments/assets/27cce9dd-7f88-48ef-9982-9cd293a025f8" />

---

<img width="689" height="240" alt="image" src="https://github.com/user-attachments/assets/b0967af7-80bf-4670-8bcb-b90342b3b3f4" />

---

<img width="689" height="1015" alt="image" src="https://github.com/user-attachments/assets/67048f25-2486-43f5-b707-e208344dbe84" />

---

<img width="690" height="789" alt="image" src="https://github.com/user-attachments/assets/ce96f238-cba4-4363-b771-e8eaf05002e0" />

---

<img width="1189" height="593" alt="image" src="https://github.com/user-attachments/assets/62b648f1-2744-4da0-bec3-b8849cc3b04c" />

---

<img width="689" height="240" alt="image" src="https://github.com/user-attachments/assets/171b54e3-6d37-45c6-9a5c-70e438162e73" />

---

<img width="690" height="1289" alt="image" src="https://github.com/user-attachments/assets/524e97b3-0596-47b5-a8a0-86dd8e5507a0" />

---

<img width="690" height="789" alt="image" src="https://github.com/user-attachments/assets/27ade0d2-422b-482e-a299-24d0ecce3d24" />

---

<img width="1476" height="812" alt="image" src="https://github.com/user-attachments/assets/0e30295f-6e0d-4aec-8a1d-d3746967cdf5" />

---

<img width="615" height="592" alt="image" src="https://github.com/user-attachments/assets/3a34f419-aa90-43f2-8270-213b9d36586e" />

---

## 💭 최종 회고 및 성찰

### 잘된 점

**파이프라인 설계의 일관성**: Model A → B → C로 이어지는 연쇄 예측 구조에서 각 모델의 출력이 다음 모델의 입력으로 유기적으로 흐르도록 설계하여 단일 추론 흐름이 명확하게 유지되었습니다.

**Data Leakage 원천 차단**: Walk-Forward OOF 기법을 통해 신재생 발전량 피처를 오염 없이 생성한 부분은 실무에서 자주 간과되는 설계 원칙을 프로젝트 초기부터 내재화했다는 점에서 의의가 있습니다.

**학술 근거 기반 시뮬레이션**: 실물 EV 데이터 확보의 물리적 한계를 7편의 논문(Heliyon, SAE, Energies, IEEE, Applied Energy, arXiv, Nature Energy)의 정량 수치로 극복한 접근법은 도메인 지식과 데이터 공학의 접점을 잘 보여줍니다.

### 아쉬운 점과 배운 것

**Model A CV Fold 7 이상치** : Fold 7에서 Val Loss가 ~140으로 급등하는 이상 패턴이 확인됩니다. 특정 시계열 구간(계절 전환, 이벤트)에 대해서 추가적으로 Fold-aware 전처리 또는 이상 기간 별도 처리 검토가 필요합니다.

**오차 전파 문제**: Pipeline A→B의 MAPE(7.28%)가 Model B 단독 MAPE(4.28%)보다 약 3%p 높게 나타났습니다. 연쇄 예측 구조에서 상위 모델의 오차가 하위 모델로 증폭되는 구조적 한계를 수치로 실감했으며, 오차 전파를 완충하는 불확실성 전달 메커니즘의 필요성을 인식하였습니다.

**합성 데이터의 한계**: Model C의 R²=0.9939라는 높은 수치는 실물 데이터가 아닌 물리 공식 기반 합성 데이터에 기인합니다. 실제 BMS CAN 데이터가 투입되면 성능이 달라질 수 있으므로 현장 검증이 반드시 선행되어야 합니다.

**제주 계통의 특수성**: 마이너스 SMP 구간은 데이터 분포 불균형 문제를 수반하며, 이를 단순 이진 분류 타겟으로 취급하는 것의 한계를 경험하였습니다.

---

## 🔭 향후 발전 계획

**실물 EV 데이터 피드백 루프 구축**  
현대차그룹 제주 V2G 시범서비스(아이오닉9·EV9 55대, 2024.12~) 실차 충방전 로그·SoC 변화 데이터를 Model C 재학습에 활용하여 공조 부하·효율 저하 계수 파라미터를 파인튜닝하는 능동 학습 루프를 구성합니다.

**오차 전파 완충 — 확률론적 파이프라인 전환**  
Model A의 7분위수 예측 분포를 Model B로 전달하는 **Monte Carlo 오차 전파 체계**를 도입하여 연쇄 예측의 불확실성을 구간 추정으로 전달합니다.

**마이너스 SMP 정밀 분류 고도화**  
`is_negative_smp` 이진 분류 성능 향상을 위해 Focal Loss 및 클래스 가중치 조정을 적용하고, 신재생 침투율(renewable_pen) 임계 기반의 도메인 룰 기반 앙상블을 병행합니다.

**실시간 배치 추론 파이프라인**  
기상청 단기 예보 API, KPX 전력거래소 데이터 피드를 연동하여 매 시간 자동 재학습 및 배치 추론을 수행하는 MLOps 파이프라인(Airflow 또는 Prefect)을 구축합니다.

**플릿 스케일 알림 시스템**  
현재 개별 EV 단위 시나리오 분석을 수십~수백 대 플릿 전체로 확장하고, V2G·DR 프로그램 연계를 통해 EV 배터리를 가상발전소(VPP) 자원으로 발전시킵니다.

**OOF 신재생 예측 모델 주기 갱신**  
`renewable_forecast` 품질 유지를 위해 Walk-Forward 검증 창을 최신 시점 데이터로 슬라이딩 업데이트하는 자동화 스케줄러를 구성합니다.

---
