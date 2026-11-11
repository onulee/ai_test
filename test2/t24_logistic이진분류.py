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
from scipy.special import expit

# 데이터 생성 - Species,Weight,Length,Diagonal,Height,Width
fish = pd.read_csv('data/fish_logistic.csv')

# Species 중복제거 출력
print(pd.unique(fish['Species']))


# df[df.columns[1:]]   # ['Species', 'Weight', 'Length', 'Diagonal', 'Height', 'Width']
data = fish[['Weight','Length','Diagonal','Height','Width']]
label = fish['Species']  # 데이터 만들기


# train데이터, test데이터 분리
train_data,test_data,train_label,test_label = train_test_split(data,label,random_state=42)


# 로지스틱회귀 먼저 이진분류를 한 후 다중분류를 함. 로지스틱 회귀는 0-1사이의 값으로 변경
# 데이터 전처리 스케일 조정, 정규화 작업
ss = StandardScaler()
ss.fit(train_data)
train_scaled = ss.transform(train_data)
test_scaled = ss.transform(test_data)


# 이진분류일때 구성을 먼저 알아봄. Bream,Smelt는 True, 그외 False
bream_smelt_index = (train_label == 'Bream') | (train_label =='Smelt')
bream_smelt_index2 = (test_label == 'Bream') | (test_label =='Smelt')
print(bream_smelt_index)
train_bream_smelt = train_scaled[bream_smelt_index]  # Bream.Smelt 데이터만 추출(33개)
label_bream_smelt = train_label[bream_smelt_index] # Bream.Smelt 데이터만 추출(33개)
test_bream_smelt = test_scaled[bream_smelt_index2]  # test_data
tlabel_bream_smelt = test_label[bream_smelt_index2] # test_label


# 로지스틱 회귀 사용 -  이진분류일때 확률형태로 보여지기 위한 함수, 다중분류는 소프트맥스 함수 적용
lr = LogisticRegression()
lr.fit(train_bream_smelt,label_bream_smelt)
print(lr.predict(train_bream_smelt[:5])) 
print(lr.predict_proba(train_bream_smelt[:5])) # 도미:0, 빙어:1 # 클래스별 확률을 반환


# 선형회귀와 같이, 기울기(가중치), Y절편이 나타남 특성이 5개 - 무게,길이,대각선,높이,두께
print(lr.coef_,lr.intercept_)    


# z = -0.404X무게 - 0.576X길이 - 0.663X대각선 - 0.013X높이 - 0.732X두께 - 2.161
# z값을 출력함.
decisions = lr.decision_function(train_bream_smelt[:5]) # z값 : 각 데이터의 결정값 계산
print(decisions) # 5개 z값 출력


# 양성값만 출력 predict_proba의 1의 값을 출력함.  
# [0.00240145 0.97264817 0.00513928 0.01415798 0.00232731]
from scipy.special import expit
print(expit(decisions))  # expit : 시그모이드함수를 적용시킴 0-1사이의 값으로 출력


# 정확도
# train_bream_smelt = train_scaled[bream_smelt_index]  # Bream.Smelt 데이터만 추출(33개)
# label_bream_smelt = train_label[bream_smelt_index] # Bream.Smelt 데이터만 추출(33개)
# test_bream_smelt = test_scaled[bream_smelt_index2]  # test_data
# tlabel_bream_smelt = test_label[bream_smelt_index2] # test_label

# 정확도
score = lr.score(train_bream_smelt,label_bream_smelt)
score2 = lr.score(test_bream_smelt,tlabel_bream_smelt)
print('정답률 : ',score) 
print('정답률2 : ',score2) 
