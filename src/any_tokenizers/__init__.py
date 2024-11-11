from .base import register_auto_tokenizer, AutoAnyTokenizer
from .titok import TiTokImageTokenizer


register_auto_tokenizer(
    TiTokImageTokenizer,
    [
        "yucornetto/tokenizer_titok_l32_imagenet",
        "yucornetto/tokenizer_titok_b64_imagenet",
        "yucornetto/tokenizer_titok_s128_imagenet"
    ]
)