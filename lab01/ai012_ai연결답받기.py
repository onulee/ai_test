#### 검색에 따른 한번에 답이 넘어와 출력됨. 실시간 스트리밍은 안됨


# 환경 설정과 API 호출에 필요한 모듈을 가져옵니다.
import os
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI

# 소스 파일의 위치를 기준으로 과정 기본 디렉터리를 계산합니다.
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)

# 실제 키를 출력하지 않고 존재 여부만 검사합니다.
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("기본 디렉터리의 .env에 OPENAI_API_KEY를 설정해 주세요.")

# 환경 변수의 API 키로 OpenAI 클라이언트를 생성합니다.
client = OpenAI(api_key=api_key)


#-----------------------------

# 입력 직후 공백을 정리해 검증과 요청에 같은 값을 사용합니다.
topic = input("수업 주제를 입력하세요: ").strip()

if not topic:
    print("수업 주제는 비워 둘 수 없습니다.")
    raise SystemExit(1)

#-----------------

try:
    # 역할과 출력 원칙은 instructions에, 현재 주제는 input에 분리합니다.
    response = client.responses.create(
        model="gpt-5.6",
        instructions=(
            "당신은 창의적인 수업 설계 도우미입니다. "
            "준비물, 진행 방법, 기대 효과를 모두 포함하고 "
            "학생이 이해하기 쉬운 한국어로 5문장 이내에 답합니다."
        ),
        input=f"'{topic}' 주제로 15분 안에 할 수 있는 활동 한 가지를 제안해 주세요.",
        max_output_tokens=1000,
    )
except openai.APIConnectionError as error:
    print(f"OpenAI API에 연결하지 못했습니다:{error}")
    raise SystemExit(1) from error
except openai.RateLimitError as error:
    print(f"요청 한도를 확인해 주세요:{error}")
    raise SystemExit(1) from error
except openai.APIError as error:
    print(f"OpenAI API 요청에 실패했습니다:{error}")
    raise SystemExit(1) from error

#-------------------------------

print("[ 확인 ] ",response)

# 최종 텍스트를 가져와 앞뒤의 공백과 줄바꿈을 정리합니다.
result = response.output_text.strip()
if not result:
    raise RuntimeError("생성된 활동 아이디어가 비어 있습니다.")

print("\n[추천 활동]")
print(result)