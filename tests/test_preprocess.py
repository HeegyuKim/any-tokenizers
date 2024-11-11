from src.any_tokenizers.base.utils import ImagePreprocessConfig, load_and_preprocess_images

image = "tests/images/ILSVRC2012_val_00008636.png"
config = ImagePreprocessConfig(
    max_size=128,
    crop_size=128,
)
images = load_and_preprocess_images([image], config=config, return_tensor="pt")
print(images.shape)