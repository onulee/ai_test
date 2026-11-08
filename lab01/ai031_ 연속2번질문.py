# 2번의 대화를 수행하면서, 두 번째 요청에서 첫 번째 응답의 맥락을 이어가는 예제

import os
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI


# 소스 파일의 위치를 기준으로 과정 기본 디렉터리의 .env를 불러옵니다.
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)

# 실제 키를 출력하지 않고 값의 존재 여부만 검사합니다.
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("기본 디렉터리의 .env에서 OPENAI_API_KEY를 찾을 수 없습니다.")

client = OpenAI(api_key=api_key)

# previous_response_id를 사용해도 자동으로 이어지지 않으므로 매 턴 재사용합니다.
ASSISTANT_INSTRUCTIONS = (
    "당신은 창의적인 수업 설계 비서입니다. "
    "학습자 수준과 수업 시간을 고려해 구체적이고 안전한 활동을 제안합니다."
)

# 2번의 요청을 수행하면서, 두 번째 요청에서 첫 번째 응답의 맥락을 이어가는 예제
# 1번째 요청과 2번째 요청을 수행하는 함수를 정의
# 1번째 요청에는 previous_id가 없음.
# 매개변수가 없으면 None으로 초기화
def create_turn(user_input: str, previous_id: str | None = None):
    """한 턴의 요청을 전송하고 텍스트가 있는 Response를 반환합니다."""
    # 딕셔너리 전개연산자 **request_options
    # "{},{},{}".format(*arr) #리스트 전개연산자
    request_options: dict[str, object] = {
        "model": "gpt-5.6",
        "instructions": ASSISTANT_INSTRUCTIONS,
        "input": user_input,
        "max_output_tokens": 600,
        # "previous_response_id":previous_id, 이전정보 가지고 감.
    }

    # 첫 턴에는 이전 ID가 없고, 후속 턴에만 직전 Response ID를 전달합니다.
    if previous_id is not None:
        request_options["previous_response_id"] = previous_id

    try:
        response = client.responses.create(**request_options)
    except openai.APIError as error:
        print(f"OpenAI API 요청에 실패했습니다: {error}")
        raise SystemExit(1) from error

    if not response.output_text.strip():
        raise RuntimeError("API 응답에 표시할 텍스트가 없습니다.")

    return response

#----- 1번째 요청

first_request = input("첫 번째 수업 설계 요청을 입력하세요: ").strip()
if not first_request:
    raise ValueError("첫 번째 요청을 입력해 주세요.")

first_response = create_turn(first_request)

print("\n[첫 답변]")
print(first_response.output_text)
print(f"첫 Response ID: {first_response.id}")


# ---- 2번째 요청

follow_up = input("\n후속 요청을 입력하세요: ").strip()
if not follow_up:
    raise ValueError("후속 요청을 입력해 주세요.")

second_response = create_turn(
    follow_up,
    previous_id=first_response.id,  # 맥락을 유지하기 위해 추가
)

print("\n[후속 답변]")
print(second_response.output_text)
print(f"연결한 Response ID: {first_response.id}")
print(f"새 Response ID: {second_response.id}")