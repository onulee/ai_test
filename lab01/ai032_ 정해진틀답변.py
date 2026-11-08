# 기업형 LLM 작성시 리턴 형태를 지정해서 받을수 있도록 구성
# 모델이 자연어로 답하면 후처리가 어려움.
# 미리 정의한 LessonPlan 구조로 응답 받기
import os
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# 정해진 데이터 구조로 리턴받아야 정리가 됨.
# Field를 사용해 각 필드에 대한 설명과 제약 조건을 지정
# ge: 크거나같음 greater than or equal to, 
# le: 작거나같음 less than or equal to, min_length, max_length
class Activity(BaseModel):
    title: str = Field(description="활동 이름")
    minutes: int = Field(ge=5, le=30, description="5분 이상 30분 이하의 활동 시간")
    materials: list[str] = Field(description="준비물 목록")
    steps: list[str] = Field(
        min_length=2,
        max_length=4,
        description="순서대로 수행할 2~4개의 활동 단계",
    )
    safety_note: str = Field(description="안전과 개인정보 보호 안내")

# Activity 모델을 포함하는 LessonPlan 모델 정의
# Activity 상위개념
# BaseModel : Pydantic LLM이 반환한 데이터가 저으이한 구조와 맞는지 검증
class LessonPlan(BaseModel):
    grade: str = Field(description="학습 대상 학년")
    topic: str = Field(description="수업 주제")
    objective: str = Field(description="관찰 가능한 학습 목표")
    activities: list[Activity] = Field(
        min_length=2,
        max_length=2,
        description="정확히 두 개의 수업 활동",
    )
    exit_ticket: str = Field(description="학습 확인을 위한 마무리 질문")


# 소스 파일의 위치를 기준으로 과정 기본 디렉터리의 .env를 불러옵니다.
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("기본 디렉터리의 .env에서 OPENAI_API_KEY를 찾을 수 없습니다.")

client = OpenAI(api_key=api_key)

# 사용자 입력을 모더레이션하고 위험 신호 여부를 반환
# -> bool 안넣어도 됨. 타입이 무엇인지 명시적으로 표현
def is_flagged(text: str) -> bool:
    """사용자 입력을 모더레이션하고 위험 신호 여부를 반환합니다."""
    try:
        # 모더레이션 API를 호출하여 입력 텍스트를 검사
        moderation = client.moderations.create(
            model="omni-moderation-latest",
            input=text,
        )
    except openai.APIError as error:
        raise RuntimeError("입력 안전 검사를 완료하지 못했습니다.") from error

    return moderation.results[0].flagged   # 내부 정책을 위반하는지 판단

# 입력값 확인
def read_safe_request(prompt: str) -> str:
    """빈 입력과 모더레이션 결과를 확인한 안전한 요청만 반환합니다."""
    user_request = input(prompt).strip()
    if not user_request:
        raise ValueError("학년과 수업 주제를 입력해 주세요.")

    if is_flagged(user_request): # 모더레이션 호출
        print("안전 안내: 교육 목적에 맞는 안전한 요청으로 다시 작성해 주세요.")
        raise SystemExit(0)

    return user_request


#--------  시작 --------------------

try:
    first_request = read_safe_request(
        "첫 번째 학년과 수업 주제를 입력하세요: "
    )
except (RuntimeError, ValueError) as error:
    print(f"입력 검사 오류: {error}")
    raise SystemExit(1) from error

print("첫 번째 입력 안전 검사: 통과")

# 한 줄의 공통 기준을 여러 번 배치해 캐시 가능한 길이의 고정 접두사를 만듭니다.
CACHE_POLICY_LINE = (
    "학습 목표와 활동은 직접 연결하고, 학생이 관찰 가능한 행동으로 목표를 작성합니다. "
    "활동은 도입, 탐구, 공유, 정리의 흐름을 따르며 학습자 수준과 수업 시간을 고려합니다. "
    "준비물은 학교에서 쉽게 구할 수 있는 것으로 제한하고 개인정보, 저작권, 신체 안전을 "
    "함께 점검합니다. 교사는 결과를 그대로 사용하지 않고 학급 상황에 맞게 검토합니다."
)

# 1024개 이상해야 가능해서 길이를 길게 만듬.
CACHE_REFERENCE = "\n".join(
    f"공통 설계 기준{index:02d}:{CACHE_POLICY_LINE}"
    for index in range(1, 25)
)

# 1024토큰 이상이어야 캐시가 가능하므로, 공통 설계 기준을 여러 번 반복하여 길이를 늘립니다.
DEVELOPER_GUIDE = f"""
당신은 학교 수업 설계 전문가입니다.
반드시 LessonPlan 스키마를 따르고 정확히 두 개의 활동을 작성합니다.
각 활동은 5분 이상 30분 이하이며, 전체 활동 시간은 가능하면 45분 이내로 구성합니다.
입력과 관련 없는 정보를 임의로 만들지 말고 교육적으로 안전한 대안을 제시합니다.

[공통 수업 설계 참고 자료]
{CACHE_REFERENCE}
""".strip()

CACHE_KEY = "lab03-lesson-planner-v1"


def build_cached_input(user_request: str) -> list[dict[str, object]]:
    """고정 개발자 접두사 뒤에 가변 사용자 요청을 배치합니다."""
    return [
        {
            "type": "message",
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": DEVELOPER_GUIDE,
                    # 이 블록 끝까지를 재사용할 명시적 캐시 접두사로 지정합니다.
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": user_request,
                }
            ],
        },
    ]



#  client.responses.create() : 자연어로 응답, 후처리 필요
#  client.responses.parse() : 구조화된 LessonPlan 객체를 생성
def create_lesson_plan(user_request: str):
    """안전한 요청을 구조화 수업안으로 생성합니다."""
    try:
        response = client.responses.parse( # 구조화된 형태로 받고 싶을때
            model="gpt-5.6",
            input=build_cached_input(user_request),
            text_format=LessonPlan,  # parse일때 사용가능 - 자연어로 만들지말고,구조에 맞게 지침
            max_output_tokens=1200,
            prompt_cache_key=CACHE_KEY, 
            prompt_cache_options={"mode": "explicit"}, #캐시를 명시적으로 사용하도록 지정
        )
    except openai.APIError as error:
        raise RuntimeError("구조화 수업안 생성에 실패했습니다.") from error

    plan = response.output_parsed  # LessonPlan 구조에 맞게 리턴됨.
    if plan is None:
        raise RuntimeError("구조화된 수업안을 만들지 못했습니다.")

    return response, plan


def print_lesson_plan(label: str, plan: LessonPlan) -> None:
    """검증된 LessonPlan 객체를 사람이 읽기 쉬운 형식으로 표시합니다."""
    print(f"\n[{label}]")
    print(f"수업 대상: {plan.grade}")
    print(f"수업 주제: {plan.topic}")
    print(f"학습 목표: {plan.objective}")

    for index, activity in enumerate(plan.activities, start=1):
        print(f"\n활동{index}: {activity.title}")
        print(f"활동 시간: {activity.minutes}분")
        print(f"준비물: {', '.join(activity.materials)}")

        for step_number, step in enumerate(activity.steps, start=1):
            print(f"진행{step_number}: {step}")

        print(f"안전 안내: {activity.safety_note}")

    print(f"\n마무리 질문: {plan.exit_ticket}")


# 캐시 사용량 확인
def get_cache_metrics(response) -> tuple[int, int]:
    """응답에서 캐시 읽기와 쓰기 토큰을 안전하게 가져옵니다."""
    input_details = getattr(response.usage, "input_tokens_details", None)
    if input_details is None:
        return 0, 0

    cached_tokens = getattr(input_details, "cached_tokens", 0) or 0
    cache_write_tokens = getattr(input_details, "cache_write_tokens", 0) or 0
    return cached_tokens, cache_write_tokens


# 첫번째 요청
try:
    first_response, first_plan = create_lesson_plan(first_request)
except RuntimeError as error:
    print(f"처리 오류: {error}")
    raise SystemExit(1) from error

print_lesson_plan("첫 번째 구조화 수업안", first_plan)

first_cached_tokens, first_cache_write_tokens = get_cache_metrics(first_response)
print("\n[첫 번째 요청 캐시 사용량]")
print(f"캐시 읽기 토큰: {first_cached_tokens}")
print(f"캐시 쓰기 토큰: {first_cache_write_tokens}")



try:
    second_request = read_safe_request(
        "\n두 번째 학년과 수업 주제를 입력하세요: "
    )
    second_response, second_plan = create_lesson_plan(second_request)
except (RuntimeError, ValueError) as error:
    print(f"처리 오류: {error}")
    raise SystemExit(1) from error

print("두 번째 입력 안전 검사: 통과")
print_lesson_plan("두 번째 구조화 수업안", second_plan)

second_cached_tokens, second_cache_write_tokens = get_cache_metrics(second_response)
print("\n[두 번째 요청 캐시 사용량]")
print(f"캐시 읽기 토큰: {second_cached_tokens}")
print(f"캐시 쓰기 토큰: {second_cache_write_tokens}")

print("\n[캐시 비교]")
print(f"첫 번째 요청 캐시 읽기: {first_cached_tokens}")
print(f"첫 번째 요청 캐시 쓰기: {first_cache_write_tokens}")
print(f"두 번째 요청 캐시 읽기: {second_cached_tokens}")
print(f"두 번째 요청 캐시 쓰기: {second_cache_write_tokens}")

if second_cached_tokens > 0:
    print("캐시 확인: 두 번째 요청이 고정 접두사를 캐시에서 읽었습니다.")
else:
    print(
        "캐시 점검: 두 번째 캐시 읽기가 0입니다. "
        "접두사, 캐시 키, 캐시 지점과 실행 간격을 확인해 주세요."
    )