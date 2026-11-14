import os
from pathlib import Path
from dotenv import load_dotenv

# Path(__file__) : 현재 실행 중인 Python 소스 파일의 경로
# .resolve() : 실제 파일 경로를 반환
# BASE_DIR = Path(__file__).resolve().parent.parent
# ENV_FILE = BASE_DIR / ".env"
# # .env 파일을 읽어 환경변수로 등록
# load_dotenv(ENV_FILE)

# .env 파일의 설정값을 읽습니다.
load_dotenv()

# 환경변수 값 가져오기
api_key = os.getenv("OPENAI_API_KEY")
print("API Key: " + api_key)

id = os.getenv("id")
pw = os.getenv("pw")
print("ID: " + id)
print("Password: " + pw)


