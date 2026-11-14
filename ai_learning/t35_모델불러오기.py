import joblib
# 모델 불러오기
model = joblib.load("model.joblib")

# input 데이터 입력
in_data = input('length, height, width 를 입력하세요.>> (예:40,11.73,7.22)')
in_data = in_data.split(',')
input_data = list(map(float,in_data))
print('입력 데이터 : ', input_data)
# 예측
result = model.predict( [input_data] )
print("예측 무게:", result[0])
