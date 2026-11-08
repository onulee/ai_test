import base64  #이미지 파일을 base64로 인코딩하기 위해 필요
import os
import time
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# 소스 파일의 위치를 기준으로 과정 기본 디렉터리와 lab04 경로를 계산합니다.
LAB_DIR = Path(__file__).resolve().parent
BASE_DIR = LAB_DIR.parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("기본 디렉터리의 .env에서 OPENAI_API_KEY를 찾을 수 없습니다.")

# API 키의 실제 값을 출력하지 않고 클라이언트에만 전달합니다.
client = OpenAI(api_key=api_key)


MODEL = "gpt-5.6"
ALLOWED_EFFORTS = {"low", "medium", "high"}
ALLOWED_DETAILS = {"auto", "low", "high", "original"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024

SYSTEM_INSTRUCTIONS = (
    "당신은 중고등학교 과학 관찰 활동을 돕는 조교입니다. "
    "이미지에서 보이지 않는 내용이나 사람의 신원, 감정, 건강 상태 같은 "
    "민감한 속성을 추측하지 않습니다."
)

ANALYSIS_PROMPT = """
이미지를 다음 형식으로 분석해 주세요.
[관찰 사실]
- 화면에서 직접 확인되는 대상, 색상, 모양, 위치, 상태만 작성합니다.
[가능한 해석]
- 관찰 사실에 근거한 해석만 작성하고 확정적인 표현을 피합니다.
[추가 확인]
- 더 정확한 판단에 필요한 정보나 촬영 조건을 질문으로 작성합니다.
흐리거나 가려져 확인할 수 없는 내용은 '확인 어려움'이라고 표시해 주세요.
""".strip()

def validate_options(effort: str, detail: str) -> tuple[str, str]:
    normalized_effort = effort.strip().lower()
    normalized_detail = detail.strip().lower()

    # 추론 수준 검사
    if normalized_effort not in ALLOWED_EFFORTS: #{"low", "medium", "high"}
        raise ValueError("추론 수준은 low, medium, high 중에서 선택해 주세요.")
    if normalized_detail not in ALLOWED_DETAILS:
        raise ValueError("세부 수준은 auto, low, high, original 중에서 선택해 주세요.")

    return normalized_effort, normalized_detail

# 추론 토큰 수를 가져오는 함
def extract_reasoning_tokens(response: object) -> int:
    # 사용량이나 세부 정보가 없을 때도 0을 반환하도록 안전하게 접근합니다.
    usage = getattr(response, "usage", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return getattr(output_details, "reasoning_tokens", 0) or 0


def analyze_image(
    image_source: str,
    effort: str,
    detail: str,
) -> tuple[str, float, int]:
    effort, detail = validate_options(effort, detail)
    started_at = time.perf_counter()


    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": effort},
        instructions=SYSTEM_INSTRUCTIONS,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": ANALYSIS_PROMPT},
                    {
                        "type": "input_image",
                        "image_url": image_source,
                        "detail": detail,
                    },
                ],
            }
        ],
        # 실습 이미지에 대한 응답을 서버 측 대화 상태로 보관하지 않습니다.
        store=False,
    )


    elapsed = time.perf_counter() - started_at


    if response.status != "completed":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", "알 수 없는 이유")
        raise RuntimeError(f"응답이 완료되지 않았습니다: {reason}")


    result_text = response.output_text.strip()
    if not result_text:
        raise RuntimeError("완료된 응답에서 분석 결과 텍스트를 찾을 수 없습니다.")


    reasoning_tokens = extract_reasoning_tokens(response)
    return result_text, elapsed, reasoning_tokens






def validate_web_url(image_url: str) -> str:
    cleaned_url = image_url.strip()
    if not cleaned_url:
        raise ValueError("이미지 URL을 입력해 주세요.")
    if any(character.isspace() for character in cleaned_url):
        raise ValueError("URL에는 공백을 포함할 수 없습니다.")


    parsed = urlparse(cleaned_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("http 또는 https로 시작하는 완전한 이미지 URL을 입력해 주세요.")
    if parsed.username or parsed.password:
        raise ValueError("사용자 이름이나 비밀번호가 포함된 URL은 사용할 수 없습니다.")


    return cleaned_url






def resolve_local_path(raw_path: str) -> Path:
    cleaned_path = raw_path.strip().strip('"')
    if not cleaned_path:
        raise ValueError("로컬 이미지 경로를 입력해 주세요.")


    candidate = Path(cleaned_path).expanduser()
    if not candidate.is_absolute():
        candidate = LAB_DIR / candidate


    return candidate.resolve()




def detect_image_mime(image_bytes: bytes) -> str:
    # 파일 확장자가 아니라 이미지의 대표적인 바이트 서명을 확인합니다.
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if (
        len(image_bytes) >= 12
        and image_bytes.startswith(b"RIFF")
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"


    raise ValueError("실제 형식이 JPEG, PNG 또는 WEBP인 이미지를 사용해 주세요.")




def to_data_url(image_path: Path) -> str:
    if not image_path.is_file():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")


    image_size = image_path.stat().st_size
    if image_size == 0:
        raise ValueError("빈 이미지 파일은 사용할 수 없습니다.")
    if image_size > MAX_IMAGE_BYTES:
        raise ValueError("이 실습에서는 10MB 이하의 이미지를 사용해 주세요.")


    image_bytes = image_path.read_bytes()
    mime_type = detect_image_mime(image_bytes)
    encoded = base64.b64encode(image_bytes).decode("ascii")


    return f"data:{mime_type};base64,{encoded}"








def main() -> None:
    try:
        mode = input("입력 방식(url/file): ").strip().lower()
        effort = input("추론 수준(low/medium/high, 기본 medium): ").strip().lower()
        detail = input(
            "세부 수준(auto/low/high/original, 기본 high): "
        ).strip().lower()


        effort = effort or "medium"
        detail = detail or "high"


        if mode == "url":
            image_source = validate_web_url(input("이미지 URL: "))
        elif mode == "file":
            image_path = resolve_local_path(input("로컬 이미지 경로: "))
            image_source = to_data_url(image_path)
        else:
            raise ValueError("입력 방식은 url 또는 file 중에서 선택해 주세요.")


        result, elapsed, reasoning_tokens = analyze_image(
            image_source=image_source,
            effort=effort,
            detail=detail,
        )


        print(f"\n분석 결과:\n{result}")
        print(f"\n입력 방식: {mode}")
        print(f"추론 수준: {effort}")
        print(f"이미지 세부 수준: {detail}")
        print(f"소요 시간: {elapsed:.2f}초")
        print(f"추론 토큰: {reasoning_tokens}")
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"입력 오류: {error}")
    except RuntimeError as error:
        print(f"처리 오류: {error}")
    except OpenAIError:
        # 상세 API 오류에 민감한 입력 정보가 포함될 수 있으므로 일반 메시지만 표시합니다.
        print("요청 오류: 이미지 분석 요청을 완료하지 못했습니다.")
    except KeyboardInterrupt:
        print("\n사용자가 실행을 중단했습니다.")


if __name__ == "__main__":
    main()


