from ..base import BaseImageTokenizer, BaseImageGenerator
from ..base.utils import load_and_preprocess_images, ImagePreprocessConfig
from typing import List, Union
from PIL.Image import Image, fromarray
import numpy as np
import torch
import torchvision.transforms as transforms
import os
from torchvision.transforms.functional import to_pil_image


COSMOS_MODELS = {
    "DI": [
        "nvidia/Cosmos-Tokenizer-DI8x8",
        "nvidia/Cosmos-Tokenizer-DI16x16",
    ],
    "CI": [
        "nvidia/Cosmos-Tokenizer-CI8x8",
        "nvidia/Cosmos-Tokenizer-CI16x16",
    ],
    "DV": [
        "nvidia/Cosmos-Tokenizer-DV4x8x8",
        "nvidia/Cosmos-Tokenizer-DV8x8x8",
        "nvidia/Cosmos-Tokenizer-DV8x16x16",
    ],
    "CV": [
        "nvidia/Cosmos-Tokenizer-CV4x8x8",
        "nvidia/Cosmos-Tokenizer-CV8x8x8",
        "nvidia/Cosmos-Tokenizer-CV8x16x16",
    ],
}


class BaseCosmosTokenizer(BaseImageTokenizer, BaseImageGenerator):
    
    def __init__(self, model, device: str = None, **kwargs):
        self.model = model
        self.device = device

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, device: str = None, load_generator=True, **kwargs):
        from .image_lib import ImageTokenizer
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(pretrained_model_name_or_path)
        encoder = ImageTokenizer(
            checkpoint_enc=os.path.join(model_dir, "encoder.jit"),
            checkpoint_dec=os.path.join(model_dir, "decoder.jit") if load_generator else None,
            device=device,
            ).eval()
        
        # (latent,) = encoder.encode(input_tensor)
        return cls(encoder, device, **kwargs)
    

class CosmosDITokenizer(BaseCosmosTokenizer):
    @torch.no_grad()
    def encode(self, x: Union[str, Image], config: ImagePreprocessConfig, **kwargs):
        return self.batch_encode([x], config, **kwargs)[0]
    
    @torch.no_grad()
    def batch_encode(self, x: List[Union[str, Image]], config: ImagePreprocessConfig = ImagePreprocessConfig(), **kwargs):
        images = load_and_preprocess_images(x, config=config)
        images = images.to(torch.bfloat16).to(self.device)
        codes, _ = self.model.encode(images)
        return codes
    
    @torch.no_grad()
    def decode(self, codes: Union[np.ndarray, torch.Tensor], **kwargs):
        if codes.ndim == 2:
            if isinstance(codes, np.ndarray):
                codes = torch.tensor(codes).to(self.device)

            return self.batch_decode(codes.unsqueeze(0), **kwargs)[0]
        
        return to_pil_image(self.model.decode(codes).float().cpu()[0])
    
    @torch.no_grad()
    def batch_decode(self, codes: Union[np.ndarray, torch.Tensor], **kwargs):
        return [to_pil_image(x.float()) for x in self.model.decode(codes).cpu()]


class CosmosCITokenizer(BaseCosmosTokenizer):
    @torch.no_grad()
    def encode(self, x: Union[str, Image], config: ImagePreprocessConfig, **kwargs):
        return self.batch_encode([x], config, **kwargs)[0]
    
    
    @torch.no_grad()
    def batch_encode(self, x: List[Union[str, Image]], config: ImagePreprocessConfig = ImagePreprocessConfig(), **kwargs):
        images = load_and_preprocess_images(x, config=config)
        images = images.to(torch.bfloat16).to(self.device)
        (codes,)= self.model.encode(images)
        return codes
    
    @torch.no_grad()
    def decode(self, codes: Union[np.ndarray, torch.Tensor], **kwargs):
        if codes.ndim == 3:
            if isinstance(codes, np.ndarray):
                codes = torch.tensor(codes).to(self.device)
                
            return self.batch_decode(codes.unsqueeze(0), **kwargs)[0]
        
        return to_pil_image(self.model.decode(codes).float().cpu()[0])
    
    @torch.no_grad()
    def batch_decode(self, codes: Union[np.ndarray, torch.Tensor], **kwargs):
        return [to_pil_image(x.float()) for x in self.model.decode(codes).cpu()]