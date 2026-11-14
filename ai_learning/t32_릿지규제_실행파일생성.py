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
import joblib
from sklearn.pipeline import make_pipeline

# 길이, 높이, 폭 데이터 가져오기
df = pd.read_csv('test2/data/perch_full.csv')

perch_weight = [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
     1000.0, 1000.0]

# 1. 훈련 데이터와 테스트 데이터 분리
train_data, test_data, train_label, test_label = \
    train_test_split(df, perch_weight, random_state=42)

# 2. 파이프라인 생성
model = make_pipeline(
    PolynomialFeatures(degree=5, include_bias=False),
    StandardScaler(),
    Ridge(alpha=0.1)
)

# 3. 모델 학습
model.fit(train_data, train_label)

# 4. 훈련 및 테스트 점수 확인
print("훈련 점수:", model.score(train_data, train_label))
print("테스트 점수:", model.score(test_data, test_label))

# 5. 모델 저장
joblib.dump(model, "model.joblib")

print("모델 저장 완료!")