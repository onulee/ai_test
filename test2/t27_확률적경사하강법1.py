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
from sklearn.linear_model import SGDClassifier
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


# 확률적 경사하강법 함수사용
sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
sc.fit(train_scaled,train_label)  #fit을 첫번째 훈련
print(sc.score(train_scaled,train_label))
print(sc.score(test_scaled,test_label))


# 확률적 경사하강법 2번째 사용
# fit 함수를 사용하면 기울기(가중치),Y절편을 모두 버리고 다시 설정함.
# partial_fit은 기존의 기울기(가중치).Y절편을 가지고 다시 훈련을 진행
sc.partial_fit(train_scaled,train_label)  # fit을 두번째 훈련
print(sc.score(train_scaled,train_label)) 
print(sc.score(test_scaled,test_label))