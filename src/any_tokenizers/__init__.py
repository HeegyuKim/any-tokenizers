from .base import register_auto_tokenizer, AutoAnyTokenizer
from .titok import TiTokImageTokenizer
from .cosmos import (
    CosmosDITokenizer, 
    CosmosCITokenizer,
    COSMOS_MODELS
)

register_auto_tokenizer(
    TiTokImageTokenizer,
    [
        "yucornetto/tokenizer_titok_l32_imagenet",
        "yucornetto/tokenizer_titok_b64_imagenet",
        "yucornetto/tokenizer_titok_s128_imagenet"
    ]
)


register_auto_tokenizer(
    CosmosDITokenizer,
    COSMOS_MODELS["DI"]
)
register_auto_tokenizer(
    CosmosCITokenizer,
    COSMOS_MODELS["CI"]
)