# Source Code 폴더

## 📋 용도
재사용 가능한 Python 모듈과 함수들을 저장하는 폴더입니다.

## 📁 예상 파일 구조
```
src/
├── data/                        # 데이터 처리 모듈
│   ├── __init__.py
│   ├── preprocessing.py         # 전처리 함수
│   ├── feature_engineering.py   # 특성 엔지니어링
│   └── data_loader.py          # 데이터 로더
├── models/                      # 모델 관련 모듈
│   ├── __init__.py
│   ├── xgboost_model.py        # XGBoost 모델
│   ├── ensemble_model.py       # 앙상블 모델
│   └── transformer_model.py    # Transformer 모델
├── utils/                       # 유틸리티 함수
│   ├── __init__.py
│   ├── metrics.py              # 평가 지표
│   ├── visualization.py        # 시각화 함수
│   └── config.py               # 설정 파일
├── experiments/                 # 실험 관련
│   ├── __init__.py
│   ├── hyperparameter_tuning.py
│   └── model_comparison.py
└── README.md                    # 이 파일
```

## 🔧 주요 모듈 설명

### data/
- **preprocessing.py**: 텍스트 정제, 수학 기호 정규화
- **feature_engineering.py**: 특성 추출 및 생성
- **data_loader.py**: 데이터 로딩 및 검증

### models/
- **xgboost_model.py**: XGBoost 모델 클래스
- **ensemble_model.py**: 앙상블 모델 구현
- **transformer_model.py**: Transformer 기반 모델

### utils/
- **metrics.py**: MAP@3, 정확도 등 평가 지표
- **visualization.py**: 혼동 행렬, 성능 그래프
- **config.py**: 하이퍼파라미터 및 설정

## 📝 코딩 규칙
1. **모듈화**: 기능별로 분리
2. **문서화**: docstring 작성
3. **테스트**: 단위 테스트 포함
4. **타입 힌트**: Python 타입 힌트 사용

## 🚀 사용 방법
```python
from src.data.preprocessing import txt_clean
from src.models.xgboost_model import XGBoostModel
from src.utils.metrics import calculate_map3_score
```

---
*생성일: 2025년 1월 21일* 