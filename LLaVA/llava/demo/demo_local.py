import torch
import gradio as gr
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.utils import disable_torch_init
import re
import time

# 모델 경로 설정
MODEL_PATH = "/home/aikusrv04/aiku/small_korean_vlm/checkpoints/lora_merged/llava-hyperclovax-korean-ocr-culture-augmented"
MODEL_BASE = None
MODEL_NAME = get_model_name_from_path(MODEL_PATH)

# 모델 불러오기
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH, MODEL_BASE, MODEL_NAME
)

device = model.device
disable_torch_init()

# conv mode 설정
if "llama-2" in MODEL_NAME.lower():
    conv_mode = "llava_llama_2"
elif "mistral" in MODEL_NAME.lower():
    conv_mode = "mistral_instruct"
elif "v1.6-34b" in MODEL_NAME.lower():
    conv_mode = "chatml_direct"
elif "v1" in MODEL_NAME.lower():
    conv_mode = "llava_v1"
elif "mpt" in MODEL_NAME.lower():
    conv_mode = "mpt"
else:
    conv_mode = "llava_v0"

conv_template = conv_templates[conv_mode].copy()

# 🔁 대화 함수 정의
def llava_chat(message, history):
    # 이미지 가져오기
    if message.get("files"):
        image_path = message["files"][0]
        image = Image.open(image_path).convert("RGB")
        question = message.get("text", "이 이미지에 대해 설명해줘!")
    else:
        return "❗ 이미지를 먼저 업로드해주세요!"

    qs = question
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # conv 초기화 및 메시지 구성
    conv = conv_template.copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 이미지 전처리
    image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
    image_size = [image.size]

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    # 응답 생성
    with torch.inference_mode():
        start = time.time()
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_size,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=512,
            use_cache=True,
            stop_strings=["<|endofturn|>", "<|im_end|>"],
            tokenizer=tokenizer,
        )
        end = time.time()

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # return outputs + f"\n\n🕒 처리 시간: {end - start:.2f}초"
    return outputs

# 🖥️ Gradio 인터페이스
demo = gr.ChatInterface(
    fn=llava_chat,
    type="messages",
    multimodal=True,
    textbox=gr.MultimodalTextbox(
        file_types=["image"],
        file_count="single",
        sources=["upload"],
        label="이미지 업로드 + 질문 입력"
    ),
    title="Small Korean VLM 멀티모달 챗봇",
    description="이미지를 업로드하고 텍스트로 질문해 보세요."
)

demo.launch(debug=True, share=True)