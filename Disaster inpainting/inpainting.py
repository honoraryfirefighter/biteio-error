import sys
import torch
import io
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler

# 추가된 경로 (필요에 따라 수정 가능)
sys.path.append('/home/etri/workspace/minji/system/clipseg_repo')
from models.clipseg import CLIPDensePredT

# 재난 유형에 따른 영어 단어 매핑
disaster_type_to_english = {
    '지진': 'earthquake',
    '지반침하': 'land subsidence',
    '싱크홀': 'sinkhole',
    '토석류': 'mudslide',
    '홍수': 'flood',
    '폭풍해일': 'storm surge'
}

# Segment prompt 설정 함수
def get_segment_prompt(disaster_type):
    ground_related = ['지진', '지반침하', '싱크홀', '토석류']
    water_related = ['홍수', '폭풍해일']
    if disaster_type in ground_related:
        return 'ground'
    elif disaster_type in water_related:
        return 'ocean'
    return 'unknown'

# 마스크 확장 함수
def expand_mask_based_on_alert_intensity(processed_mask, alert_intensity):
    kernel_size = 20
    if alert_intensity == '주의보':
        kernel_size = 40
    elif alert_intensity == '경보':
        kernel_size = 60
    expanded_mask = torch.nn.functional.max_pool2d(processed_mask.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=kernel_size//2).squeeze()
    return (expanded_mask - expanded_mask.min()) / (expanded_mask.max() - expanded_mask.min())

def apply_inpainting(image, disaster_type, alert_intensity):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 이미지 데이터 타입 확인 및 변환
    if isinstance(image, io.BytesIO):
        image.seek(0)  # 스트림의 시작 부분으로 포인터 이동
        image = Image.open(image).convert('RGB')  # BytesIO를 PIL Image 객체로 변환
    elif isinstance(image, str):
        image = Image.open(image).convert('RGB')  # 파일 경로에서 이미지 로드 및 변환

    tensor_image = transforms.ToTensor()(image).unsqueeze(0).to(device)

    # 모델 설정
    model_path = '/home/etri/workspace/minji/system/clipseg_weights/clipseg_weights/rd64-uni.pth'
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64).to(device)
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    # Stable Diffusion 인페인팅 파이프라인 설정
    model_dir = "stabilityai/stable-diffusion-2-inpainting"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler").to(device)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_dir, scheduler=scheduler, revision="fp16", torch_dtype=torch.float32).to(device)

    # ClipSeg를 사용하여 마스크 생성
    segment_prompt = get_segment_prompt(disaster_type)
    with torch.no_grad():
        preds = model(tensor_image, [segment_prompt])[0]
    processed_mask = torch.special.ndtr(preds[0][0]).to(device)
    expanded_mask = expand_mask_based_on_alert_intensity(processed_mask, alert_intensity)
    stable_diffusion_mask = transforms.ToPILImage()(expanded_mask.cpu())

    # 인페인팅 프롬프트 설정
    english_disaster_type = disaster_type_to_english.get(disaster_type, 'disaster')
    inpainting_prompt = f"{english_disaster_type} has occurred"

    # 이미지 인페인팅을 수행합니다.
    generator = torch.Generator(device=device).manual_seed(77)
    result_image = pipe(prompt=inpainting_prompt, guidance_scale=7.5, num_inference_steps=60, generator=generator, image=tensor_image, mask_image=stable_diffusion_mask).images[0]

    return result_image
