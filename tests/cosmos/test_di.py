from src.any_tokenizers import AutoAnyTokenizer, ImagePreprocessConfig
import os


model_path = "nvidia/Cosmos-Tokenizer-DI8x8"
tokenizer = AutoAnyTokenizer.from_pretrained(model_path, device="cuda")

files = os.listdir("tests/images/")
files = [os.path.join("tests/images/", f) for f in files]
print(files)

config = ImagePreprocessConfig(
    max_size=128,
    crop_size=128,
)
tokens = [tokenizer.encode(file, config) for file in files]
print(tokens)
print([x.shape for x in tokens])

for file, img in zip(files, tokens):
    img = tokenizer.decode(img)
    img.save("tests/images/" + "reconstructed_" + model_path.split("/")[-1] + "_" + os.path.basename(file))