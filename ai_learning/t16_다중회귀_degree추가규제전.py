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

# poly 추가되는 방법 설명
# poly.fit([[2,3]])
# # 1(bias),2,3,2**2,2*3,3**2 -> 형태로 구성
# # [[1. 2. 3. 4. 6. 9.]]
# train_poly = poly.transform([[2,3]])


poly.fit(train_data)
train_poly = poly.transform(train_data)
# 특성이 3개(length,height,width)에서 9개 늘어남, 행은 42개 (42,9)
# test_data에도 변환을 해야 함.
test_poly = poly.transform(test_data)
# train_poly구조 출력
print(train_poly.shape)


# 각 특성이 어떻게 만들어 졌는지 연산 형태를 알려줌 : 특성(변수) 구성
print(poly. get_feature_names_out())    # 변경됨.


# LinearRegression 모델 적용
lr = LinearRegression()
lr.fit(train_poly,train_label)
print('정확도 : ',lr.score(train_poly,train_label))
print('정확도 : ',lr.score(test_poly,test_label))

