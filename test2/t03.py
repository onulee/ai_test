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

# 산점도그래프 그리기
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('길이')
plt.ylabel('무게')
plt.show()

# 도미, 빙어 데이터 합치기
length = bream_length+smelt_length
weight = bream_weight+smelt_weight

# 2개의 데이터를 1개의 형태로 묶음 - 앞쪽35, 뒤쪽14
data = [[l, w] for l, w in zip(length, weight)]
label = [1]*35 + [0]*14
# print(fish_data) # 데이터 확인
# sklearn 라이브러리 사용
train_data,test_data,train_label,test_label = train_test_split(data,label)


clf = KNeighborsClassifier()
# clf = svm.SVC()
clf.fit(train_data,train_label)
result =clf.predict(test_data)
print('결과값 : ',result) 
score = metrics.accuracy_score(result,test_label)
print('정답률 : ',score)

data1 = input('길이와 무게를 입력하세요(예: 30 600) : ')

# 공백을 기준으로 문자열 분리 후 실수형 변환
length1, weight1 = map(float, data1.split())

# 예측할 데이터를 2차원 배열로 변환
input_data = [[length1, weight1]]

# KNN 모델로 예측
predict = clf.predict(input_data)

# 예측 결과 출력
if predict[0] == 1:
    print('예측 결과 : 도미입니다.')
else:
    print('예측 결과 : 빙어입니다.')
    
# 산점도그래프 그리기
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(length1, weight1, marker='^')
plt.xlabel('길이')
plt.ylabel('무게')
plt.show()    
