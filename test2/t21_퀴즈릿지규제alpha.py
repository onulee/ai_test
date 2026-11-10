# 그래프관련 import
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False

import pandas as pd
import numpy as np
from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error   # 평균절대값오차 오차범위
from sklearn.preprocessing import StandardScaler  # 정규화클래스
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('test2/data/MultipleLinearRegressionData.csv')

# 장소 뺀 컬럼
data = df.iloc[:, :-2].values
label = df.iloc[:, -1].values

# 장소분리
place = df['place'].to_numpy().reshape(-1,1)
# 원핫인코딩 객체 생성
encoder = OneHotEncoder(sparse_output=False,drop='first')
# 원핫인코딩 적용
result = encoder.fit_transform(place)
# 숫자 데이터 + 원핫인코딩 결합
total_data = np.concatenate((data, result), axis=1)
total_data

# 1. 데이터 변환
data = np.array(total_data, dtype=float)
label = np.array(label, dtype=float)

# 2. 훈련데이터, 테스트데이터 분리
train_data, test_data, train_label, test_label = train_test_split(
    data, label, random_state=42, test_size=0.2
)

print('훈련데이터:', train_data.shape)
print('테스트데이터:', test_data.shape)

# 3. 다항 특성 자동 생성
poly = PolynomialFeatures(degree=2, include_bias=False)
poly.fit(train_data)
train_poly = poly.transform(train_data)
test_poly = poly.transform(test_data)

print('다항 특성 훈련 데이터:', train_poly.shape)
print('다항 특성 테스트 데이터:', test_poly.shape)

# 특성 이름 확인
print('생성된 특성:')
print(poly.get_feature_names_out())

# 4. 표준화
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 5. 릿지 회귀 - 규제 강도 탐색
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_scaled, train_label)
    train_score.append(ridge.score(train_scaled, train_label))
    test_score.append(ridge.score(test_scaled, test_label))

# 6. alpha별 성능 그래프
plt.plot(np.log10(alpha_list), train_score, marker='o')
plt.plot(np.log10(alpha_list), test_score, marker='o')
plt.xlabel('log10(alpha)')
plt.ylabel('R2 Score')
plt.legend(['Train', 'Test'])
plt.grid()
plt.show()

# 7. 최적 alpha 선정
best_index = np.argmax(test_score)
best_alpha = alpha_list[best_index]
print('최적 alpha:', best_alpha)

# 8. 최적 alpha로 최종 모델 생성
ridge = Ridge(alpha=best_alpha)
ridge.fit(train_scaled, train_label)

print('\n[최종 모델 성능]')
print('훈련 정확도:', ridge.score(train_scaled, train_label))
print('테스트 정확도:', ridge.score(test_scaled, test_label))

# 9. 예측 및 오차 확인
result = ridge.predict(test_scaled)

print('\n실제 정답:', test_label)
print('예측 결과:', result)

mae = mean_absolute_error(test_label, result)
print('평균 절대 오차(MAE):', mae)
