from typing import List, Union, Tuple
import torch
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as Image
import math
from dataclasses import dataclass


@dataclass
class ImagePreprocessConfig:
    """
    Image preprocess configuration.
    Args:
        max_size (int): Maximum size of the image.
        crop_size (int): Crop size of the image.
        multiple_of (int): Multiple of the image size.
    """
    max_size: int = None
    min_size: int = None
    crop_size: int = None
    multiple_of: int = 1
    image_type: str = "RGB"

    def to_transform(self):
        """
        Convert the configuration to a transform function.
        Returns:
            callable: Transform function.
        """
        transform = []
        if self.max_size:
            transform.append(ConditionalResizeAndCrop(
                self.max_size, 
                self.min_size,
                self.crop_size, 
                self.multiple_of))
        transform.append(transforms.ToTensor())
        return transforms.Compose(transform)

def load_and_preprocess_images(
        images: List[Union[str, Image.Image]],
        config: ImagePreprocessConfig = ImagePreprocessConfig(),
        return_tensor: str = "pt",
        ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
    """
    Load and preprocess images for the model.
    Args:
        images (List[Union[str, Image]]): List of image paths or PIL images.
    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor]]: Preprocessed images.
    """

    images = [Image.open(img) if isinstance(img, str) else img for img in images]
    
    if config.image_type:
        images = [image.convert(config.image_type) for image in images]

    preprocess_function = config.to_transform()
    images = [preprocess_function(img) for img in images]
    
    if return_tensor == "pt":
        return torch.stack(images)
    elif return_tensor == "np":
        return torch.stack(images).numpy()
    else:
        return tuple(images)
    

    


class ConditionalResizeAndCrop:
    def __init__(self, max_size, min_size=None, crop_size=None, multiple_of=1):
        self.max_size = max_size
        self.crop_size = crop_size
        self.min_size = min_size
        self.multiple_of = multiple_of
    
    def make_multiple(self, size):
        # 주어진 수를 multiple_of의 배수로 올림
        return math.ceil(size / self.multiple_of) * self.multiple_of
    
    def __call__(self, img):
        w, h = img.size
        
        # 리사이즈가 필요한 경우
        if w > self.max_size or h > self.max_size:
            ratio = self.max_size / float(max(w, h))
            # n의 배수로 맞추기
            new_w = self.make_multiple(int(w * ratio))
            new_h = self.make_multiple(int(h * ratio))
            img = transforms.Resize((new_h, new_w))(img)
        else:
            # 리사이즈가 필요없더라도 n의 배수로 맞추기
            new_w = self.make_multiple(w)
            new_h = self.make_multiple(h)
            if new_w != w or new_h != h:
                img = transforms.Resize((new_h, new_w))(img)
        
        if self.min_size:
            w, h = img.size
            if w < self.min_size or h < self.min_size:
                ratio = self.min_size / float(min(w, h))
                new_w = self.make_multiple(int(w * ratio))
                new_h = self.make_multiple(int(h * ratio))
                img = transforms.Resize((new_h, new_w))(img)

        # 크롭이 필요한 경우
        if self.crop_size:
            w, h = img.size
            if w > self.crop_size or h > self.crop_size:
                # 크롭 사이즈도 n의 배수로 맞추기
                crop_size = self.make_multiple(self.crop_size)
                return transforms.CenterCrop(crop_size)(img)
        
        return img
    

def video_to_pil_images(video_path):
    import cv2
    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)
    
    # 비디오를 읽을 수 없는 경우 예외 처리
    if not cap.isOpened():
        raise Exception(f"Error: Could not open video file {video_path}")
    
    pil_images = []
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        
        # 더 이상 프레임이 없으면 종료
        if not ret:
            break
            
        # OpenCV BGR을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # numpy 배열을 PIL Image로 변환
        pil_image = Image.fromarray(rgb_frame)
        
        # 리스트에 추가
        pil_images.append(pil_image)
    
    # 비디오 캡처 객체 해제
    cap.release()
    
    return pil_images



def load_and_preprocess_videos(
        video_paths: Union[str, List[str]],
        config: ImagePreprocessConfig = ImagePreprocessConfig(),
        return_tensor: str = "pt",
        ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
    """
    Load and preprocess video for the model.
    Args:
        video_paths (Union[str, List[str]]): Video path or list of video paths.
        preprocess_function (callable): Preprocess function.
        image_type (str): Image type.
        return_tensor (str): Return tensor type.
    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor]]: Preprocessed video.
    """
    video_images = []
    for video_path in video_paths:
        pil_images = video_to_pil_images(video_path)
        video_images.append(load_and_preprocess_images(pil_images, config, return_tensor))

    return torch.stack(video_images)