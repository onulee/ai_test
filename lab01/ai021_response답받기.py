# 환경 설정, 시간 측정, API 오류 처리에 필요한 모듈을 가져옵니다.
import os
import time
from pathlib import Path


import openai
from dotenv import load_dotenv
from openai import OpenAI


# 소스 파일의 위치를 기준으로 과정 기본 디렉터리와 .env 경로를 계산합니다.
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


# 지정한 .env의 값을 현재 Python 프로세스의 환경 변수로 불러옵니다.
load_dotenv(ENV_FILE)




# 실제 키를 출력하지 않고 값의 존재 여부만 검사합니다.
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("기본 디렉터리의 .env에서 OPENAI_API_KEY를 찾을 수 없습니다.")


# 검증한 API 키로 OpenAI 클라이언트를 생성합니다.
client = OpenAI(api_key=api_key)


# 입력 직후 공백을 정리해 검증과 요청에 같은 값을 사용합니다.
ingredients = input("냉장고 속 재료를 쉼표로 구분해 입력하세요: ").strip()
if not ingredients:
    raise ValueError("재료를 한 가지 이상 입력해 주세요.")




started_at = time.perf_counter()


try:
    # stream을 생략하면 완성된 Response 객체가 반환됩니다.
    response = client.responses.create(
        model="gpt-5.6",
        instructions=(
            "당신은 남은 재료를 알뜰하게 활용하는 셰프입니다. "
            "요리 이름, 4단계 조리법, 대체 가능한 재료를 한국어로 제안합니다."
        ),
        input=f"사용 가능한 재료: {ingredients}",
        max_output_tokens=600,
    )
except openai.APIConnectionError as error:
    print(f"OpenAI API에 연결하지 못했습니다: {error}")
    raise SystemExit(1) from error
except openai.RateLimitError as error:
    print(f"요청 한도와 프로젝트 사용량을 확인해 주세요: {error}")
    raise SystemExit(1) from error
except openai.APIError as error:
    print(f"OpenAI API 요청에 실패했습니다: {error}")
    raise SystemExit(1) from error


finished_at = time.perf_counter()


# 출력 Item의 고정 위치를 가정하지 않고 집계된 최종 텍스트를 가져옵니다.
result = response.output_text.strip()
if not result:
    raise RuntimeError("API 응답에 표시할 텍스트가 없습니다.")


print("\n[완성된 답변]")
print(result)
print(f"\n전체 소요 시간: {finished_at - started_at:.2f}초")

