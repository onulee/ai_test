from django.shortcuts import render
from pathlib import Path
import joblib

app_name=''
def index(request):
    # Django 프로젝트 root 경로
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # 모델 불러오기
    model = joblib.load(BASE_DIR / "model.joblib")

    # in_data = input('length, height, width 를 입력하세요.>> (예:40,11.73,7.22)')
    in_data = '40,11.73,7.22'
    in_data = in_data.split(',')
    input_data = list(map(float,in_data))
    print('입력 데이터 : ',input_data)

    # 예측
    result = model.predict([input_data])
    print("예측 무게:", result[0])
    context = {'result':result[0]}
    return render(request,'index.html',context)
