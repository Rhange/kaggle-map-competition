# Feature CV XGBoost 참고 자료

## 📋 개요
- **원작자**: 외부 공유자 (Kaggle 커뮤니티)
- **목적**: MAP - Charting Student Math Misunderstandings 대회를 위한 XGBoost 모델
- **평가 지표**: MAP@3 (Mean Average Precision at 3)

## 🔧 주요 기능

### 1. 데이터 전처리
- **텍스트 결합**: `QuestionText + MC_Answer + StudentExplanation`
- **수학 기호 정규화**:
  - 분수 표현: `\frac{a}{b}` → `FRAC_a_b`
  - 분수 표현: `a/b` → `FRAC_a_b`
- **텍스트 정제**: 특수문자 제거, 공백 정규화, 소문자 변환

### 2. 특성 엔지니어링
#### 수학적 특성 추출
```python
features = {
    'frac_count': 분수 개수,
    'number_count': 숫자 개수,
    'operator_count': 연산자 개수 (+, -, *, /, =),
    'multiply_sign_count': 곱셈 기호 개수 (*, ×, ·, times),
    'power_count': 거듭제곱 개수 (^, **, squared, cubed)
}
```

#### 길이 기반 특성
- `mc_answer_len`: 객관식 답안 길이
- `explanation_len`: 학생 설명 길이
- `question_len`: 문제 길이
- `explanation_to_question_ratio`: 설명/문제 비율

#### 텍스트 처리
- **TF-IDF 벡터화**: 텍스트 특성 추출
- **Lemmatization**: WordNetLemmatizer 사용
- **수학적 표현 정규화**: 수식 패턴 인식

### 3. 모델링
- **알고리즘**: XGBoost
- **교차 검증**: StratifiedKFold
- **평가 지표**: MAP@3
- **예측**: 상위 3개 `Category:Misconception` 조합

### 4. 시각화
- **혼동 행렬**: Red 색상 테마
- **성능 평가**: 정확도 및 MAP@3 점수

## 📁 파일 구조
```
references/feature-cv-xgboost/
├── README.md                    # 이 파일
├── feature-cv-xgboost.py       # Python 스크립트
└── feature-cv-xgboost.ipynb    # Jupyter 노트북
```

## 🚀 사용 방법
1. Kaggle 환경에서 실행
2. 데이터 경로 설정:
   - `/kaggle/input/map-charting-student-math-misunderstandings/train.csv`
   - `/kaggle/input/map-charting-student-math-misunderstandings/test.csv`
   - `/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv`

## 📊 성능
- **OOF MAP@3 Score**: 계산 완료
- **모델**: XGBoost with Cross-Validation
- **특성**: 텍스트 + 수학적 특성 + 길이 기반 특성

## 💡 주요 인사이트
1. **텍스트 결합**: 문제, 답안, 설명을 하나로 결합하여 컨텍스트 활용
2. **수학적 표현 정규화**: 수학 기호를 모델이 인식할 수 있는 형태로 변환
3. **MAP@3 평가**: 상위 3개 예측의 정확도를 평가하는 것이 대회 목표에 적합

## 🔄 개선 가능한 부분
1. **하이퍼파라미터 튜닝**: Optuna 등 사용
2. **앙상블 모델**: 여러 모델 조합
3. **추가 특성**: 문법적 특성, 의미적 특성
4. **데이터 증강**: 텍스트 증강 기법 적용

---
*참고 자료: 외부 공유자 (Kaggle 커뮤니티)* 