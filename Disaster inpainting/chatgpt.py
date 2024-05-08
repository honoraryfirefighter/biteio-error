from openai import OpenAI
import os

# Set the API key from an environment variable
os.environ["OPENAI_API_KEY"] = "챗지피티 api 키"

# Create an OpenAI client instance
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_disaster_alert(alert_text):
    instruction = f"""
    재난 텍스트 입력을 재난 발생 위치, 종류, 강도 및 발생 시간 정보로 변환하세요. 분류할 수 없는 항목은 None으로 출력하세요. 모든 출력은 한국어로 해야 합니다.

    [예시 1]
    입력: 오늘 15시 20분경 유성구 용산동 화장품 공장에서 화재 사고 발생. 인근 주민은 외출을 자제하고 안전에 유의하세요. [유성구]
    출력:
    재난 발생 위치: 유성구 용산동 화장품 공장
    재난 종류: 화재
    재난 강도: None
    재난 발생 시간: 15시 20분

    [예시 2]
    입력: 오늘 05:20 갑천 대전시(원촌교) 호우로 인한홍수주의보 발령, 안전에 유의하시기 바랍니다.
    출력:
    재난 발생 위치: 대전시 갑천 원촌교
    재난 종류: 호우, 홍수
    재난 강도: 주의보
    재난 발생 시간: 05:20

    [사용자 입력]
    입력: {alert_text}
    """
    
    # 챗 GPT를 사용하여 분석 요청
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": alert_text}
        ]
    )

    # 응답에서 필요한 정보 추출
    response_content = response.choices[0].message.content
    parsed_output = {}  # 최종 반환될 사전

    # 응답 내용 분석
    lines = response_content.strip().split('\n')
    for line in lines:
        if '재난 발생 위치:' in line:
            parsed_output['재난 발생 위치'] = line.split(': ')[1]
        elif '재난 종류:' in line:
            parsed_output['재난 종류'] = line.split(': ')[1]
        elif '재난 강도:' in line:
            parsed_output['재난 강도'] = line.split(': ')[1] if line.split(': ')[1] else None
        elif '재난 발생 시간:' in line:
            parsed_output['재난 발생 시간'] = line.split(': ')[1]

    # 필요한 정보가 없으면 None으로 채움
    keys = ['재난 발생 위치', '재난 종류', '재난 강도', '재난 발생 시간']
    for key in keys:
        if key not in parsed_output:
            parsed_output[key] = None

    return parsed_output
