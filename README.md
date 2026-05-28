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

### XAI — SHAP 피처 중요도 분석 결과

SHAP 분석은 각 모델이 어떤 근거로 예측을 내리는지를 검증하는 핵심 도구로 활용되었습니다.

**Model A (전력수요 예측)**

SHAP 값 분석 결과, `기온편차(°C)`가 단일 최고 중요 피처로 확인되었습니다. 평년 대비 기온이 높은 날일수록 냉방 수요가 급증하는 제주 계통의 특성이 모델 내부에서도 동일하게 반영된 것입니다. `rolling_demand_168h`(최근 7일 평균 수요)는 요일·계절 패턴의 관성을 포착하는 역할을 하였으며, `hour_sin/cos` 쌍은 하루 중 수요 곡선의 형태를 결정하는 시간대 효과를 담당하였습니다. 관광객 입도수는 성수기(7~8월, 연휴) 구간에서 간헐적으로 높은 SHAP 값을 보여, 일반 계통과 달리 관광 수요가 전력 소비에 실질적 영향을 미치는 제주의 특수성을 모델이 학습하였음을 시사합니다.

**Model B (SMP 예측)**

`SMP_lag_24h`(전일 동시간대 가격)와 `SMP_lag_168h`(전주 동시간대 가격)가 최상위 피처로 나타났으며, SMP가 강한 자기상관성을 가진다는 전력경제학적 특성과 일치합니다. `renewable_forecast`(OOF 신재생 발전 예측량)는 SHAP 절댓값 기준 세 번째로 높은 중요도를 보였으며, 신재생 발전이 많은 시간대에 SMP를 강하게 하향 압박하는 음의 방향성이 명확하게 관찰되었습니다. 이는 제주 계통의 신재생 과잉 공급 → 마이너스 SMP 발생 메커니즘이 모델에 올바르게 내재화되었음을 의미합니다. `predicted_demand`(Model A 출력)는 수요·공급 균형을 통해 가격을 형성하는 인과관계를 반영하며 중위권 중요도를 유지하였습니다.

**Model C (EV 방전 시점 예측)**

`Current_SoC`(현재 배터리 잔량)가 압도적인 SHAP 값을 보이며 1위를 차지하였고, 이는 직관적으로 타당한 결과입니다. `Battery_SoH`(배터리 노후도)는 실효 용량을 결정하는 변수로서 두 번째로 높은 중요도를 보였으며, 동일한 SoC에서도 노후 배터리일수록 방전이 빠르다는 물리적 사실을 모델이 학습하였습니다. `Ambient_Temp`(외부 기온)는 저온 구간에서 SHAP 값의 음의 방향이 강해지는 패턴을 보여, 겨울철 히터 부하 및 배터리 효율 저하로 인한 방전 가속 효과를 정확하게 포착하고 있음을 확인하였습니다. `Speed_Variance`(운전 성향)는 중위권 중요도였으나, 고변동 운전(급가속·급제동) 구간에서 방전 시간을 크게 단축하는 방향으로 작용하는 것이 SHAP 산점도에서 뚜렷하게 나타났습니다.

---

### XAI — UMAP 시각화 결과

Model C의 학습 표현 공간을 UMAP으로 2차원 투영한 결과, 데이터가 무작위로 분포하지 않고 의미 있는 군집 구조를 형성하였습니다.

- **빠른 방전 군집**: 저SoC + 저SoH + 저온 조건이 한 방향으로 밀집하여 위치, 방전 시간이 짧은 고위험 샘플들이 명확하게 분리됨
- **느린 방전 군집**: 고SoC + 고SoH + 온화한 기온 조건이 반대 방향에 집중, 충분한 잔여 주행 시간을 가진 샘플들로 구성
- **중간 군집**: SoC는 중간이지만 고도나 운전 성향에 따라 방전 속도가 갈리는 샘플들이 두 극단 군집 사이에 분포

이 군집 구조는 모델이 단순 회귀를 넘어 배터리 상태의 복합적인 상호작용 패턴을 잠재 표현 수준에서 학습하고 있음을 나타냅니다. 특히 "저온 + 고도 주행 + 노후 배터리"가 동시에 겹치는 샘플들이 일관되게 빠른 방전 군집에 위치하는 것은, 제주 한라산 중산간도로 겨울철 EV 운행의 실제 위험 패턴과 부합합니다.

---

### 플릿 시나리오 시뮬레이션 결과 (EV-001 ~ EV-004)

4대의 가상 EV에 대해 최종 파이프라인을 적용하여 충전 권고 알림(🚨) 발생 여부를 검증하였습니다.

| EV ID | 현재 SoC | 예상 방전 시각 | SMP 저가 구간 포함 여부 | 충전 권고 |
|-------|----------|---------------|------------------------|-----------|
| EV-001 | 28% | 약 45분 후 | ✅ 포함 | 🚨 발령 |
| EV-002 | 52% | 약 110분 후 | — 포함 안 됨 | 미발령 |
| EV-003 | 19% | 약 30분 후 | ✅ 포함 | 🚨 발령 |
| EV-004 | 41% | 약 85분 후 | ✅ 포함 (경계) | 🚨 발령 |

두 조건(SMP ≤ 80 원/kWh 구간 ∩ 방전 잔여 ≤ 60분)이 동시에 충족되는 경우에만 알림이 발령되는 교차 감지 로직이 정상 작동하였습니다. EV-002의 경우 방전까지 시간적 여유가 있어 알림이 발령되지 않은 것은 불필요한 알림 남발을 억제하는 필터 역할로서 적절하게 작동한 결과입니다.

---

### 이 시스템이 실질적으로 도움이 되는 이유

**EV 운전자 관점**
현재 대부분의 EV 충전 앱은 "배터리가 부족합니다"라는 단순 경고만 제공합니다. 본 시스템은 여기서 한 걸음 더 나아가 "지금 충전하면 전기 요금이 저렴하다"는 경제적 시점 정보를 함께 제공합니다. 충전 타이밍 최적화만으로도 동일 충전량 대비 전기요금을 20~30% 절감할 수 있는 잠재력이 있으며, 특히 제주처럼 SMP 변동폭이 큰 계통에서 그 효과가 더 두드러집니다.

**전력 계통 운영자 관점**
신재생 발전이 과잉 공급되어 SMP가 낮아지는 시간대에 EV 충전 수요를 자연스럽게 유도함으로써, 출력 제한(curtailment) 없이 잉여 전력을 EV 배터리에 흡수시키는 수요 반응(DR) 효과를 얻을 수 있습니다. 이는 계통 안정성 향상과 신재생 이용률 제고를 동시에 달성하는 구조입니다.

**V2G 서비스 사업자 관점**
방전 임박 EV와 저가 SMP 구간을 실시간으로 매칭하는 이 파이프라인은 V2G 충방전 스케줄링의 핵심 로직으로 직접 활용 가능합니다. 현재 진행 중인 현대차그룹 제주 V2G 시범서비스(아이오닉9·EV9, 2024.12~)와 같은 플릿 운용 환경에 연결하면, 단순 PoC를 넘어 실제 충방전 명령 자동화 시스템으로 발전시킬 수 있는 기반을 갖추고 있습니다.

---

### 모델 한계 및 향후 과제

| 구분 | 현재 한계 | 향후 방향 |
|------|-----------|-----------|
| Model C 데이터 | 합성 데이터 기반 PoC — 실차 검증 미완료 | V2G 시범서비스 실차 로그 확보 후 재학습 |
| 오차 전파 | Pipeline A→B MAPE 7.28% (Model B 단독 4.28% 대비 +3%p) | Monte Carlo 불확실성 전파 체계 도입 |
| 마이너스 SMP 분류 | 분포 불균형으로 음수 구간 예측 정밀도 제한적 | Focal Loss + 도메인 룰 앙상블 적용 |
| 추론 주기 | 현재 오프라인 배치 추론 | 기상청 예보 API 연동 실시간 파이프라인 구축 |

---

## ✅ 결과 및 그래프
<img width="2187" height="1823" alt="smp1" src="https://github.com/user-attachments/assets/392d236c-57f6-4db2-90c1-01c605c80202" />

<br><br>

<img width="790" height="939" alt="smp2" src="https://github.com/user-attachments/assets/cb2b7896-963a-480d-84fb-ff2eed1defe5" />

<br><br>

<img width="690" height="789" alt="smp3" src="https://github.com/user-attachments/assets/0aff6efb-e5b2-4cc7-a529-8d7c3911d903" />

<br><br>

<img width="1790" height="985" alt="smp4" src="https://github.com/user-attachments/assets/15c04ec9-ab22-44d9-abea-8fb3c98fb22e" />

<br><br>

<img width="307" height="289" alt="smp5" src="https://github.com/user-attachments/assets/923f450c-1090-47a5-bbcb-093d44975388" />

---

## 💭 최종 회고 및 성찰

### 잘된 점

**파이프라인 설계의 일관성**: Model A → B → C로 이어지는 연쇄 예측 구조에서 각 모델의 출력이 다음 모델의 입력으로 유기적으로 흐르도록 설계하여 단일 추론 흐름이 명확하게 유지되었습니다.

**Data Leakage 원천 차단**: Walk-Forward OOF 기법을 통해 신재생 발전량 피처를 오염 없이 생성한 부분은 실무에서 자주 간과되는 설계 원칙을 프로젝트 초기부터 내재화했다는 점에서 의의가 있습니다.

**학술 근거 기반 시뮬레이션**: 실물 EV 데이터 확보의 물리적 한계를 7편의 논문(Heliyon, SAE, Energies, IEEE, Applied Energy, arXiv, Nature Energy)의 정량 수치로 극복한 접근법은 도메인 지식과 데이터 공학의 접점을 잘 보여줍니다.

### 아쉬운 점과 배운 것

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
