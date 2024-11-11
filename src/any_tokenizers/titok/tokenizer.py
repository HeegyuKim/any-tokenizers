from ..base import BaseImageTokenizer, BaseImageGenerator
from ..base.utils import load_and_preprocess_images
from typing import List, Union
from PIL.Image import Image, fromarray
import numpy as np
import torch
import torchvision.transforms as transforms


TITOK_TRANSFORM = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

TITOK_MODELS = [
    "yucornetto/tokenizer_titok_l32_imagenet", 
    "yucornetto/tokenizer_titok_b64_imagenet", 
    "yucornetto/tokenizer_titok_s128_imagenet"
]

class TiTokImageTokenizer(BaseImageTokenizer, BaseImageGenerator):
    def __init__(self, model, device: str = None):
        self.model = model
        self.device = device

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: str, device: str = None):
        from .modeling.titok import TiTok
        titok_tokenizer = TiTok.from_pretrained(pretrained_model_name_or_path)
        titok_tokenizer.eval()
        titok_tokenizer.requires_grad_(False)

        if device:
            titok_tokenizer.to(device)
        
        return TiTokImageTokenizer(titok_tokenizer, device)

    @torch.no_grad()
    def encode(self, x: Union[str, Image], **kwargs):
        return self.encode_batch([x], **kwargs)[0]
    
    @torch.no_grad()
    def encode_batch(self, x: List[Union[str, Image]], **kwargs):
        images = load_and_preprocess_images(x, preprocess_function=TITOK_TRANSFORM)
        images = images.to(self.device)
        codes = self.model.encode(images)[1]['min_encoding_indices'].squeeze()
        return codes
    
    @torch.no_grad()
    def decode(self, codes: Union[np.ndarray, torch.Tensor], **kwargs):
        if codes.ndim == 1:
            return self.decode_batch(codes.unsqueeze(0), **kwargs)[0]
        reconstructed_image = self.model.decode_tokens(codes)
        reconstructed_image = torch.clamp(reconstructed_image, 0.0, 1.0)
        reconstructed_image = (reconstructed_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
        reconstructed_image = fromarray(reconstructed_image)
        return reconstructed_image
    
    @torch.no_grad()
    def decode_batch(self, codes: Union[np.ndarray, torch.Tensor], **kwargs):
        reconstructed_images = self.model.decode_tokens(codes)
        reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
        reconstructed_images = (reconstructed_images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        reconstructed_images = [fromarray(image) for image in reconstructed_images]
        return reconstructed_images