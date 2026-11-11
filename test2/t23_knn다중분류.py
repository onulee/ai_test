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
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error   # 평균절대값오차 오차범위
from sklearn.preprocessing import StandardScaler  # 정규화클래스
from sklearn.preprocessing import OneHotEncoder


# 데이터 생성 - Species,Weight,Length,Diagonal,Height,Width
df = pd.read_csv('data/fish_logistic.csv')
# print(fish.head())
# Species 중복제거 출력
print(pd.unique(df['Species']))



# df[df.columns[1:]]   # ['Species', 'Weight', 'Length', 'Diagonal', 'Height', 'Width']
# 데이터 전처리
data = df[['Weight','Length','Diagonal','Height','Width']]
label = df['Species']  # 데이터 만들기
# numpy변경
# data = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
# label = fish['Species'].to_numpy()


# train데이터, test데이터 분리
train_data,test_data,train_label,test_label = train_test_split(data,label,random_state=42)


# 데이터 전처리 스케일 조정, 정규화 작업
ss = StandardScaler()
ss.fit(train_data)
train_scaled = ss.transform(train_data)
test_scaled = ss.transform(test_data)


# knn알고니즘 대입
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(train_scaled,train_label) 

# 결과값에 있는 항목을 나열해줌.
print("결과값 항목 : ",clf.classes_) #class 결과값 항목들이 어떤 것이 있는지 모두 보여줌.

# 데이터 예측
result =clf.predict(test_scaled)
print("결과값 : ",result)

# 결과값을 확률로 보여줌
proba = clf.predict_proba(test_scaled[:5]) # predict 1개의 확률값만 출력, predict_proba:모든항목별 확률값이 나옴.
# proba  # 소수점 부분이 보여 보기 힘듬
print(np.round(proba, decimals=4))


# 정확도
score = clf.score(train_scaled,train_label)
score2 = clf.score(test_scaled,test_label)
print('정답률 : ',score) 
print('정답률2 : ',score2) 


