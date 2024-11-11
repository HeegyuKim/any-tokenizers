from typing import List, Union, Tuple
import torch
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as Image
import math



def load_and_preprocess_images(
        images: List[Union[str, Image.Image]],
        preprocess_function: callable = None,
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
    if preprocess_function:
        images = [preprocess_function(img) for img in images]
    
    if return_tensor == "pt":
        return torch.stack(images)
    elif return_tensor == "np":
        return torch.stack(images).numpy()
    else:
        return tuple(images)
    

    


class ConditionalResizeAndCrop:
    def __init__(self, max_size, crop_size=None, multiple_of=1):
        self.max_size = max_size
        self.crop_size = crop_size
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
        
        # 크롭이 필요한 경우
        if self.crop_size:
            w, h = img.size
            if w > self.crop_size or h > self.crop_size:
                # 크롭 사이즈도 n의 배수로 맞추기
                crop_size = self.make_multiple(self.crop_size)
                return transforms.CenterCrop(crop_size)(img)
        
        return img
