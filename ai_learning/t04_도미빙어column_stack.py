import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
# matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus']=False
import pandas as pd
from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
print('bream_length : ',len(bream_length))
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
print('smelt_length : ',len(smelt_length))

length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# 열기준으로 2개 합치기
data = np.column_stack((length, weight)) # row_stack 행기준으로 
# 라벨값 생성
label = np.concatenate((np.ones(35), np.zeros(14)))
# sklearn 라이브러리에서 랜덤으로 실습데이터,테스트데이터 분리
train_data,test_data,train_label,test_label = train_test_split(data,label)
clf = KNeighborsClassifier()
clf.fit(train_data,train_label) 
result =clf.predict(test_data)

# 해당데이터 가장 근접한 데이터 5개의 거리와 index값 추출
distances, indexes = clf.kneighbors([[25, 150]])
# 산점도그래프 그리기 x축 0:길이 y축 1:무게
plt.scatter(train_data[:,0], train_data[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_data[indexes,0], train_data[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
score = clf.score(test_data,test_label)
print('정답률 : ',score) 


