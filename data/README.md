# Data 폴더

## 📋 용도
데이터 파일들을 저장하고 관리하는 폴더입니다.

## 📁 예상 파일 구조
```
data/
├── raw/                        # 원본 데이터
│   ├── train.csv              # 훈련 데이터
│   ├── test.csv               # 테스트 데이터
│   └── sample_submission.csv  # 제출 샘플
├── processed/                  # 전처리된 데이터
│   ├── train_processed.csv
│   ├── test_processed.csv
│   └── features.csv
├── features/                   # 특성 파일
│   ├── tfidf_features.pkl
│   ├── math_features.csv
│   └── text_features.csv
├── submissions/                # 제출 파일
│   ├── submission_v1.csv
│   ├── submission_v2.csv
│   └── best_submission.csv
└── README.md                   # 이 파일
```

## 📊 데이터 설명
- **train.csv**: 훈련 데이터 (QuestionText, MC_Answer, StudentExplanation, Category, Misconception)
- **test.csv**: 테스트 데이터 (QuestionText, MC_Answer, StudentExplanation)
- **sample_submission.csv**: 제출 형식 샘플

## 🔧 데이터 처리 단계
1. **Raw Data**: 원본 데이터 저장
2. **Processed Data**: 전처리된 데이터
3. **Features**: 추출된 특성들
4. **Submissions**: 제출 파일들

## 📝 데이터 정보 기록
- 데이터 버전
- 전처리 방법
- 특성 설명
- 데이터 품질 체크 결과

---
*생성일: 2025년 1월 21일* 