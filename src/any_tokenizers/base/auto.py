from .tokenizer import BaseAnyTokenizer

ANY_TOKENIZER_REGISTRY = {}


def register_auto_tokenizer(cls, model_paths, overwrite=False):
    for model_path in model_paths:
        if not overwrite and model_path in ANY_TOKENIZER_REGISTRY:
            raise ValueError(f"Model {model_path} is already registered")
        ANY_TOKENIZER_REGISTRY[model_path] = cls


class AutoAnyTokenizer(BaseAnyTokenizer):

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: str, **kwags):
        if pretrained_model_name_or_path not in ANY_TOKENIZER_REGISTRY:
            raise ValueError(f"Model {pretrained_model_name_or_path} is not registered")
        
        return ANY_TOKENIZER_REGISTRY[pretrained_model_name_or_path].from_pretrained(pretrained_model_name_or_path, **kwags)