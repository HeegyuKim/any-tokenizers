from src.any_tokenizers import AutoAnyTokenizer
import os


model_path = "nvidia/Cosmos-Tokenizer-CI8x8"
tokenizer = AutoAnyTokenizer.from_pretrained(model_path, device="cuda")

files = os.listdir("tests/images/")
files = [os.path.join("tests/images/", f) for f in files if "reconstructed" not in f]
print(files)

tokens = [tokenizer.encode(file) for file in files]
print(tokens)
print([x.shape for x in tokens])

for file, img in zip(files, tokens):
    img = tokenizer.decode(img)
    img.save("tests/images/" + "reconstructed_" + model_path.split("/")[-1] + "_" + os.path.basename(file))