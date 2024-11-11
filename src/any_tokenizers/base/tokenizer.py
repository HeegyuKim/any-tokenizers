from typing import List, Tuple, Union
import PIL
from PIL.Image import Image
import numpy as np
import torch
from .utils import ImagePreprocessConfig



class BaseAnyTokenizer:

    @property
    def codebook_size(self):
        raise NotImplementedError
    
    
    def encode(self, x, **kwargs):
        raise NotImplementedError
    
    def batch_encode(self, x, **kwargs):
        raise NotImplementedError
    
    
class BaseAnyGenerator:

    def decode(self, codes, **kwargs):
        raise NotImplementedError
    
    def batch_decode(self, codes, **kwargs):
        raise NotImplementedError
    


class BaseImageTokenizer(BaseAnyTokenizer):

    def encode(self, x: Union[str, Image], config: ImagePreprocessConfig = ImagePreprocessConfig(), **kwargs):
        raise NotImplementedError
    
    def batch_encode(self, x: List[Union[str, Image]], config: ImagePreprocessConfig = ImagePreprocessConfig(), **kwargs):
        raise NotImplementedError



class BaseImageGenerator(BaseAnyGenerator):

    def decode(self, codes: Union[np.ndarray, torch.Tensor], **kwargs):
        raise NotImplementedError
    
    def batch_decode(self, codes: Union[np.ndarray, torch.Tensor], **kwargs):
        raise NotImplementedError