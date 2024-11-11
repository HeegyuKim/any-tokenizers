from src.any_tokenizers import AutoAnyTokenizer, ImagePreprocessConfig
from datasets import get_dataset_config_names, load_dataset
import os
from functools import partial
import jsonlines
from tqdm.auto import tqdm

# tokenizer_name = "yucornetto/tokenizer_titok_l32_imagenet"
tokenizer_name = "nvidia/Cosmos-Tokenizer-DI16x16"
tokenizer = AutoAnyTokenizer.from_pretrained(tokenizer_name, device="cuda")
output_dir = "./kollava-instruct-cosmos-di16"
config = ImagePreprocessConfig(
    max_size=256,
    crop_size=256,
    multiple_of=128,
)

def tokenizer_images(item):
    image = tokenizer.encode(item["images"], config=config)
    item["image_token_shapes"] = image.shape
    item["images"] = image.cpu().tolist()
    item["image_tokenizer"] = tokenizer_name
    return item


dd = load_dataset("kihoonlee/KoLLaVA-Instruct-313k")
os.makedirs(output_dir, exist_ok=True)

for split, ds in dd.items():
    with jsonlines.open(os.path.join(output_dir, f"{split}.json"), "w") as f:
        for item in tqdm(ds, desc=f"kollava/{split}", position=1):
            f.write(tokenizer_images(item))