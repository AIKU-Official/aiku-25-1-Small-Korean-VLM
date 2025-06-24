import gradio as gr
import torch
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration

# device = "cuda:3" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")
device = "cpu"

# 모델과 프로세서 불러오기
processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = LlavaForConditionalGeneration.from_pretrained(
    "iamnotwhale/llava-svlm",
    torch_dtype=torch.float16
).to(device)

print("Model and processor loaded successfully!")


uploaded_image = None

def llava_chat(message, _):
    if message.get("files"):
        image_path = message["files"][0]
        # print(image_path)
        image = Image.open(image_path).convert("RGB")
        question = message.get("text", "Explain the image in detail.")
    else:
        return "Please upload the image first!"

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)

    generate_ids = model.generate(**inputs, max_new_tokens=128)
    output = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

    if "ASSISTANT:" in output:
        output = output.split("ASSISTANT:")[1].strip()

    return output

demo = gr.ChatInterface(
    fn=llava_chat,
    type="messages",
    multimodal=True,
    textbox=gr.MultimodalTextbox(file_types=["image"], file_count="single", sources=["upload"]),
    title="🐣 LLaVA 멀티모달 챗봇"
)

demo.launch(debug=True)