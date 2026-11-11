import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False

import pandas as pd
import numpy as np
from sklearn import svm,metrics
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression 
from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error   # 평균절대값오차 오차범위
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,OneHotEncoder
from scipy.special import expit,softmax


# 데이터 생성 - Species,Weight,Length,Diagonal,Height,Width
fish = pd.read_csv('data/fish_logistic.csv')
# Species 중복제거 출력
print(pd.unique(fish['Species']))

data = fish[['Weight','Length','Diagonal','Height','Width']]
label = fish['Species']  # 데이터 만들기


# train데이터, test데이터 분리
train_data,test_data,train_label,test_label = train_test_split(data,label,random_state=42)


# 데이터 전처리 스케일 조정, 정규화 작업
ss = StandardScaler()
ss.fit(train_data)
train_scaled = ss.transform(train_data)
test_scaled = ss.transform(test_data)


# 기본값 : 반복횟수 max_iter=100 설정되어 있음. 반복횟수가 너무 적다라고 경고뜸.
# 로지스틱회귀는 L2규제를 기본적용, 그런데 alpha 적용하지 않고, 분류는 C적용 , C는 꼭 대문자
# C 숫자가 올라가면 규제가 약해지고, 낮아지면 규제가 강해짐. 선형회귀 alpha와는 반대로 되어 있음
# C의 기본값은 1
lr = LogisticRegression(C=20,max_iter=1000)  # LogisticRegression() # 정확도 : 0.806722 / 0.85 - 규제가 완화필요
lr.fit(train_scaled,train_label)

# 다중분류에서는 로지스틱함수를 사용하면 확률 모든 값이 1이 안됨
# 시그모이드함수를 돌린 후 결과값의 합은 100% ->1 이 안됨. 이렇게 만들어 주는 것이 softmax함수임 
#                  [ -6.5    1.03    5.16  -2.73   3.34   0.33    -0.63 ]
# softmax 후[     0.    0.014 0.841 0.       0.136  0.007 0.003  ]  : 1이 됨. 100% 확률가능
# 지수함수를 sum으로 나눠서 모두 더함, 정규화작업, 모두 더하면 결과값이 1이 나옴.
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision,decimals=2))
# 기울기(가중치),절편
print( lr.coef_.shape, lr.intercept_.shape )
# (7,5) (7,)  # 7:클래스=['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt’] 
              # 5: 특성= 무게, 길이,대각선,높이,두께
              
              
print('정확도 : ',lr.score(train_scaled,train_label))
print('정확도 : ',lr.score(test_scaled,test_label))


# 시그모이드함수를 돌린 후 결과값의 합은 100% ->1 이 안됨. 이렇게 만들어 주는 것이 softmax함수임.
soft1 = softmax(decision,axis=1) # axis=0 열방향, axis=1 행방향
print(np.round(soft1,decimals=3))
            









