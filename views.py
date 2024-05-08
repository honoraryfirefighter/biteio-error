from django.shortcuts import render
import requests
from urllib.parse import quote
import json
from .src.main import parsed_disaster, main
import base64
from io import BytesIO
import os
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings


# Create your views here.
def search_panorama(request):
    if request.method == 'POST':
        address = request.POST.get('address')  # POST로부터 검색할 주소를 가져옴

        alert_text = parsed_disaster(address)
        print(alert_text['재난 발생 위치'])
        res = search_naver_local('강남역')
        print(res)
        result = json.dumps(search_naver_local("강남역"))
        print(result)


        return render(request, 'panorama.html', {'address': address, 'result':result, 'alert_text':alert_text })

    else:
        return render(request, 'search.html', {'error_message': '잘못된 접근입니다.'})


def search_naver_local(keyword, start=1, display=1):
    headers = {
        'X-Naver-Client-Id': '아이디 입력',
        'X-Naver-Client-Secret': '시크릿키입력'
    }

    encoded_keyword = quote(keyword)
    url = f"https://openapi.naver.com/v1/search/local.json?query={encoded_keyword}&start={start}&display={display}"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def disaster(request):

    context = {}
    return render(request, 'disaster.html')

@csrf_exempt
def disaster_img(request):
    disaster_dir = os.path.join(settings.BASE_DIR, 'static', 'disaster')
    for filename in os.listdir(disaster_dir):
        file_path = os.path.join(disaster_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                # 폴더 삭제가 필요한 경우 여기서 shutil.rmtree를 사용할 수 있습니다.
                pass
        except Exception as e:
            return HttpResponse(f"Error deleting file {filename}: {e}", status=500)

    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))  # JSON 데이터 파싱
        image_data = data.get('image')
        alert_text = data.get('alert_text')
        print(alert_text)
        if image_data.startswith('data:image/png;base64,'):
            # 접두사를 제거하여 순수한 Base64 인코딩된 데이터만 추출
            base64_encoded_data = image_data.replace('data:image/png;base64,', '')
        else:
            # 이미지 데이터가 예상한 형식이 아님
            # 적절한 오류 처리를 여기에 구현
            raise ValueError("Invalid image format")
        image_data = base64.b64decode(base64_encoded_data)
        image_data = BytesIO(image_data)

        main(alert_text, image_data)

        images_dir = os.path.join(settings.BASE_DIR, 'static', 'disaster')
        image_file = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
        return render(request, 'disaster_output.html', {'image_file':image_file})
    else : return render(request, 'panorama.html')