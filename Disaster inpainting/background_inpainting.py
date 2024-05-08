import sys
import torch
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler

# 경로 추가 (리눅스 경로 형식에 맞게 조정)
sys.path.append('/home/etri/workspace/minji/system/clipseg_repo')

from models.clipseg import CLIPDensePredT

def background_inpainting(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU가 사용 가능한 경우 CUDA로 설정
    
    # 모델 설정
    model_path = '/home/etri/workspace/minji/system/clipseg_weights/clipseg_weights/rd64-uni.pth'
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model = model.to(device)  # 모델을 GPU로 이동
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    
    # Stable Diffusion 인페인팅 파이프라인 설정
    model_dir = "stabilityai/stable-diffusion-2-inpainting"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler").to(device)  # 스케줄러를 GPU로 이동
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_dir, scheduler=scheduler, revision="fp16", torch_dtype=torch.float32)
    pipe = pipe.to(device)  # 파이프라인을 GPU로 이동

    # 이미지 로드 및 전처리
    source_image = Image.open(image_path).convert('RGB')
    tensor_image = transforms.ToTensor()(source_image).unsqueeze(0).to(device)  # 텐서를 GPU로 이동

    # ClipSeg를 사용하여 'sky' 세그멘테이션 후 마스크 생성
    clipseg_prompt = 'sky'
    with torch.no_grad():
        preds = model(tensor_image, [clipseg_prompt])[0]
    processed_mask = torch.special.ndtr(preds[0][0]).to(device)  # 처리된 마스크를 GPU로 이동
    stable_diffusion_mask = transforms.ToPILImage()(processed_mask.cpu())  # 마스크를 CPU로 이동하여 PIL 이미지로 변환

    # 'dark and cloudy sky'로 인페인팅 수행
    inpainting_prompt = "dark and cloudy sky"
    generator = torch.Generator(device=device).manual_seed(77)
    result_image = pipe(prompt=inpainting_prompt, guidance_scale=11, num_inference_steps=60, generator=generator, image=source_image, mask_image=stable_diffusion_mask).images[0]

    return result_image

