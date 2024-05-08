from PIL import Image, ImageEnhance, ImageChops
import numpy as np
import cv2

def create_rain_effect(image_input):
    # 이미지가 파일 경로인 경우 불러오기, 이미지 객체인 경우 직접 사용
    if isinstance(image_input, str):
        base_image = Image.open(image_input)
    else:
        base_image = image_input

    base_image = base_image.resize((512, 512))
    if base_image.mode != 'RGB':
        base_image = base_image.convert('RGB')

    # 검은색 배경 위에 백색 노이즈 생성
    noise = np.random.normal(loc=0, scale=1, size=(512, 512))
    noise_scaled = ((noise - noise.min()) / (noise.max() - noise.min()) * 255).astype('uint8')
    noise_image = Image.fromarray(noise_scaled)
    noise_image = noise_image.convert('L')

    # 모션 블러 적용을 위한 커널 생성 및 회전
    kernel_size = 15
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size

    angle = 45
    center = (kernel_size // 2, kernel_size // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_kernel_motion_blur = cv2.warpAffine(kernel_motion_blur, rot_mat, (kernel_size, kernel_size))

    # 모션 블러 적용
    noise_array = np.array(noise_image)
    rain_effect = cv2.filter2D(noise_array, -1, rotated_kernel_motion_blur)

    # numpy 배열을 PIL 이미지로 변환, 대비 및 밝기 조정
    rain_effect_image = Image.fromarray(rain_effect)
    contraster = ImageEnhance.Contrast(rain_effect_image)
    rain_effect_image = contraster.enhance(5.0)
    enhancer = ImageEnhance.Brightness(rain_effect_image)
    enhanced_noise_image = enhancer.enhance(0.6)
    
    if enhanced_noise_image.mode == 'L':
        enhanced_noise_image = enhanced_noise_image.convert('RGB')

    # 최종 이미지 합성
    final_image = screen_blend(enhanced_noise_image, base_image)
    return final_image

def screen_blend(top, bottom):
    """스크린 블렌드 모드를 구현하는 함수"""
    return ImageChops.invert(ImageChops.multiply(ImageChops.invert(top), ImageChops.invert(bottom)))

