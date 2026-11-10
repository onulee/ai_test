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

# 길이, 높이, 폭 데이터 가져오기
df = pd.read_csv('test2/data/perch_full.csv')

perch_weight = [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
     1000.0, 1000.0]

# 훈련데이터, 테스트데이터 분리
train_data, test_data, train_label, test_label = \
    train_test_split(df, perch_weight, random_state=42)


# 특성을 추가해서 정확도를 높임 
# 길이 -> 길이, 높이, 폭 3개 특성으로 예측
# 다항특성 자동생성 3**2 = 9개, 기본 1 제외
poly = PolynomialFeatures(degree=5,include_bias=False)


poly.fit(train_data)
train_poly = poly.transform(train_data)
# 특성이 3개(length,height,width)에서 9개 늘어남, 행은 42개 (42,9)
# test_data에도 변환을 해야 함.
test_poly = poly.transform(test_data)
# train_poly구조 출력
print(train_poly.shape)


# 각 특성이 어떻게 만들어 졌는지 연산 형태를 알려줌 : 특성(변수) 구성
print(poly. get_feature_names_out())    # 변경됨.

# 규제,정규화작업 진행
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)


# 규제 강도 찾기 : 릿지회귀 : 가중치의 제곱을 규제 (L2규제) / 라쏘회귀 : 가중치의 절대값을 규제(L1규제)
train_score = []
test_score = []
# 규제강도 찾기 : 5개 구성
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 릿지 모델을 만듭니다
    # 릿지 회귀 사용 - 규제,정규화 작업후 릿지회귀 사용함.
    ridge = Ridge(alpha=alpha)
    # 릿지 모델을 훈련합니다
    ridge.fit(train_scaled, train_label)
    # 훈련 점수와 테스트 점수를 저장합니다
    train_score.append(ridge.score(train_scaled, train_label))
    test_score.append(ridge.score(test_scaled, test_label))


# 선 그래프 
list=[1,2,3,4,5,6]
plt.plot(list, train_score)   # x좌표 눈금을 -3,-2,-1,0,1,2 -> 1간격으로 만들기 위해 log10적용
plt.plot(list, test_score)    # x좌표 눈금 -1 인 -> 0.1이 가장 좋은 정확도가 나타남
# plt.plot(np.log10(alpha_list), train_score)   # x좌표 눈금을 -3,-2,-1,0,1,2 -> 1간격으로 만들기 위해 log10적용
# plt.plot(np.log10(alpha_list), test_score)    # x좌표 눈금 -1 인 -> 0.1이 가장 좋은 정확도가 나타남
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

# # 릿지 회귀 : 가중치의 제곱을 벌칙으로 사용
# 규제정규화 alpha=0.1 적용
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled,train_label)

print('훈련 정확도 : ', ridge.score(train_scaled, train_label))
print('테스트 정확도 : ', ridge.score(test_scaled, test_label))


