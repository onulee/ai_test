## 실시간으로 응답 받음 - 한번에 넘어오는 것이 아니라 실시간을 단어를 출력

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
load_dotenv(ENV_FILE)


# 실제 키를 출력하지 않고 값의 존재 여부만 검사합니다.
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("기본 디렉터리의 .env에서 OPENAI_API_KEY를 찾을 수 없습니다.")


client = OpenAI(api_key=api_key)


ingredients = input("냉장고 속 재료를 쉼표로 구분해 입력하세요: ").strip()
if not ingredients:
    raise ValueError("재료를 한 가지 이상 입력해 주세요.")


# 요청 시간과 스트리밍 처리 상태를 각각 저장합니다.
started_at = time.perf_counter()
first_delta_at = None
pieces: list[str] = []
completed_response = None

print("\n[스트리밍 답변]")

try:
    # stream=True를 지정하면 처리 과정의 이벤트를 순서대로 받습니다.
    stream = client.responses.create(
        model="gpt-5.6",
        instructions=(
            "당신은 남은 재료를 알뜰하게 활용하는 셰프입니다. "
            "요리 이름, 4단계 조리법, 대체 가능한 재료를 한국어로 제안합니다."
        ),
        input=f"사용 가능한 재료: {ingredients}",
        max_output_tokens=600,
        stream=True,
    )

    # 이벤트 유형을 먼저 확인한 뒤 해당 이벤트에 정의된 필드만 사용합니다.
    for event in stream:
        if event.type == "response.output_text.delta":
            if first_delta_at is None:
                first_delta_at = time.perf_counter()

            pieces.append(event.delta)
            print(event.delta, end="", flush=True)


        elif event.type == "response.completed":
            # 전체 Response가 정상 완료됐음을 확인하고 완성 객체를 보관합니다.
            completed_response = event.response


except openai.APIConnectionError as error:
    print(f"\nOpenAI API에 연결하지 못했습니다: {error}")
    raise SystemExit(1) from error
except openai.RateLimitError as error:
    print(f"\n요청 한도와 프로젝트 사용량을 확인해 주세요: {error}")
    raise SystemExit(1) from error
except openai.APIError as error:
    print(f"\nOpenAI API 요청에 실패했습니다: {error}")
    raise SystemExit(1) from error


finished_at = time.perf_counter()

print()

if completed_response is None:
    raise RuntimeError("response.completed 이벤트를 받지 못해 후처리를 중단합니다.")

if first_delta_at is None:
    raise RuntimeError("표시할 텍스트 델타를 받지 못했습니다.")

# 화면에 표시한 모든 텍스트 조각을 원래 순서대로 결합합니다.
full_text = "".join(pieces)
if not full_text.strip():
    raise RuntimeError("수집한 전체 텍스트가 비어 있습니다.")

print(f"\n첫 출력 표시 시간: {first_delta_at - started_at:.2f}초")
print(f"전체 소요 시간: {finished_at - started_at:.2f}초")
print(f"수집한 전체 글자 수: {len(full_text)}")
print(f"완료 이벤트 확인: {completed_response.status == 'completed'}")
print(completed_response)
