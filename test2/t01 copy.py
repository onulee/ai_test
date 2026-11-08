from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from sklearn import svm, metrics

# 1. 이미지 열기 및 흑백 변환
img = Image.open("test2/data/seven.png").convert("L")

# 색상 반전
img = ImageOps.invert(img)

# 숫자가 있는 영역만 추출
bbox = img.getbbox()
img = img.crop(bbox)

# 비율을 유지하면서 20x20 이내로 조정 - 여백을 부여하기 위해
# LANCZOS 필터를 사용하여 고품질로 축소
img.thumbnail((20, 20), Image.Resampling.LANCZOS)

# 28x28 검은 배경 생성
new_img = Image.new("L", (28, 28), 0)

# 숫자를 중앙에 배치
x = (28 - img.width) // 2
y = (28 - img.height) // 2
new_img.paste(img, (x, y))

# 예측용 데이터 생성
data = np.array(new_img, dtype=np.float32)  # 넘파이 배열로 변환
flat_data2 = data.reshape(1, 784) / 256.0   # 1차원 배열로 변환 및 정규화

# 이미지 확인
new_img.resize((280, 280)).show()


# 6. 학습 데이터 읽기
train_csv = pd.read_csv(
    "test2/data/train.csv", header=None
)
tk_csv = pd.read_csv(
    "test2/data/t10k.csv", header=None
)

# 7. 학습 데이터 분리
X_train = train_csv.iloc[:, 1:].to_numpy(
    dtype=np.float32
) / 255.0

y_train = train_csv.iloc[:, 0].to_numpy()

X_test = tk_csv.iloc[:, 1:].to_numpy(
    dtype=np.float32
) / 255.0

y_test = tk_csv.iloc[:, 0].to_numpy()

# 8. SVM 학습
clf = svm.SVC()
clf.fit(X_train, y_train)

# 9. 테스트 데이터 예측
predict = clf.predict(X_test)



# 11. 정확도 출력
score = metrics.accuracy_score(y_test, predict)
print("정답률:", score)

# 10. 직접 만든 이미지 예측
# 예측
predict2 = clf.predict(flat_data2)
print("예측값:", predict2[0])
# 12. 각 숫자에 대한 결정 점수 확인
scores = clf.decision_function(flat_data2)
print("결정 점수:", scores)
