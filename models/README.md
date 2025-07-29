# Models 폴더

## 📋 용도
훈련된 머신러닝 모델들을 저장하는 폴더입니다.

## 📁 예상 파일 구조
```
models/
├── xgboost/                    # XGBoost 모델
│   ├── model_v1.pkl           # 첫 번째 버전
│   ├── model_v2.pkl           # 개선된 버전
│   └── config.json            # 모델 설정
├── ensemble/                   # 앙상블 모델
│   └── ensemble_v1.pkl
├── transformer/                # Transformer 모델
│   └── bert_model/
└── README.md                   # 이 파일
```

## 💾 모델 저장 형식
- **Pickle (.pkl)**: Python 객체 직렬화
- **Joblib (.joblib)**: 대용량 배열에 최적화
- **ONNX**: 프레임워크 간 호환성
- **H5**: Keras 모델

## 📝 모델 정보 기록
각 모델 파일과 함께 다음 정보를 기록:
- 모델 버전
- 훈련 날짜
- 성능 지표 (MAP@3, 정확도 등)
- 사용된 특성
- 하이퍼파라미터

---
*생성일: 2025년 1월 21일* 