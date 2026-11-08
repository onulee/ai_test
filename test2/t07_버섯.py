from sklearn import svm,metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


# 데이터 읽기---1
df = pd.read_csv('test2/data/mushroom.csv',header=None)
# 데이터 내부의 기호를 숫자로 취환---2
data = []
label = []
attr_list = []
for row_index,row in df.iterrows():  # pandas DataFrame 데이터에서 행(row)를 1개씩 가져옴.
 label.append(row[0])
 row_data = []
 for v in row[1:]:
     row_data.append(ord(v))  # ord('a') 정수 97을 반환 - 유니코드 정수로 변환하는 함수, chr() 유니코드 문자변환
 data.append(row_data) 
 
# 학습데이터,테스트데이터 나누기---3
train_data,test_data,train_label,test_label = train_test_split(data,label)
# 데이터 학습시키기---4
clf = RandomForestClassifier()
clf.fit(train_data,train_label)
# 데이터 예측하기---5
predict = clf.predict(test_data)
# 결과 테스트---6
score = metrics.accuracy_score(test_label,predict)  
report = metrics.classification_report(test_label,predict)
print("정답률 : ",score) 
print("리포트 :\n ",report) 