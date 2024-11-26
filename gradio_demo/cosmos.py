import gradio as gr
from any_tokenizers import AutoAnyTokenizer, ImagePreprocessConfig
from PIL import Image
import json
import numpy as np


# 토크나이저 초기화
model_path = "nvidia/Cosmos-Tokenizer-DI16x16"
tokenizer = AutoAnyTokenizer.from_pretrained(model_path, device="cuda")

TITOK_DEFAULT_CONFIG = ImagePreprocessConfig(
    max_size=1024,
    multiple_of=128,
)


def encode_image(image):
    # 이미지를 토큰화
    tokens = tokenizer.encode(image, TITOK_DEFAULT_CONFIG)
    return str(tokens.cpu().tolist())

def decode_tokens(tokens):
    # 토큰을 이미지로 복원
    tokens = np.array(json.loads(tokens))
    image = tokenizer.decode(tokens)
    return image

# Gradio 인터페이스 생성
with gr.Blocks() as demo:
    gr.Markdown("# 이미지 토큰화/복호화 데모")
    
    with gr.Tab("이미지 토큰화"):
        with gr.Row():
            input_image = gr.Image(type="pil", label="입력 이미지")
            output_tokens = gr.Textbox(label="생성된 토큰")
        encode_btn = gr.Button("토큰화")
        encode_btn.click(fn=encode_image, inputs=input_image, outputs=output_tokens)
    
    with gr.Tab("토큰 복호화"):
        with gr.Row():
            input_tokens = gr.Textbox(label="토큰 입력")
            output_image = gr.Image(type="pil", label="복원된 이미지")
        decode_btn = gr.Button("복호화")
        decode_btn.click(fn=decode_tokens, inputs=input_tokens, outputs=output_image)

# 데모 실행
demo.launch(share=True)