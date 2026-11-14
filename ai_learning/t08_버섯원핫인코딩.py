import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 1. CSV 파일 읽기
df = pd.read_csv(
    'test2/data/mushroom.csv',
    header=None
)


# 2. 입력 데이터와 정답 분리
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# 3. 범주형 데이터 숫자로 변환
X = pd.get_dummies(X)

print("입력 데이터 shape:", X.shape)
print("입력 데이터 shape:", X)
# 4. 학습용과 테스트용 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. 랜덤 포레스트 모델 생성
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 6. 모델 학습
clf.fit(X_train, y_train)

# 7. 예측
predict = clf.predict(X_test)

# 8. 평가
score = metrics.accuracy_score(y_test, predict)

report = metrics.classification_report(
    y_test, predict
)

print("정답률 :", score)
print("분류 리포트 :")
print(report)