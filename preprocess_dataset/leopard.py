from src.any_tokenizers import AutoAnyTokenizer
from datasets import get_dataset_config_names, load_dataset
import os
from functools import partial
import jsonlines
from tqdm.auto import tqdm

# tokenizer_name = "yucornetto/tokenizer_titok_l32_imagenet"
tokenizer_name = "nvidia/Cosmos-Tokenizer-DI16x16"
tokenizer = AutoAnyTokenizer.from_pretrained(tokenizer_name, device="cuda")
output_dir = "/data3/heegyu/leopard-instruct-cosmos-di16"

def tokenizer_images(item, subset: str):
    # return {
    #     "images": [tokenizer.encode_image(image) for image in item["images"]],
    #     "subset": subset
    # }
    images = [tokenizer.encode(image) for image in item["images"]]
    item["image_token_shapes"] = [image.shape for image in images]
    item["images"] = [image.cpu().tolist() for image in images]
    item["subset"] = subset
    item["image_tokenizer"] = tokenizer_name
    return item


for config_name in tqdm(get_dataset_config_names("wyu1/Leopard-Instruct"), desc="Tokenizing", position=0):
    dd = load_dataset("wyu1/Leopard-Instruct", config_name)
    subset_dir = os.path.join(output_dir, config_name)
    os.makedirs(subset_dir, exist_ok=True)

    for split, ds in dd.items():
        with jsonlines.open(os.path.join(subset_dir, f"{split}.json"), "w") as f:
            for item in tqdm(ds, desc=f"{config_name}/{split}", position=1):
                f.write(tokenizer_images(item, config_name))