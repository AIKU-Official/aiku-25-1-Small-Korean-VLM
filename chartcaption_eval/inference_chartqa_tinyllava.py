import os
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from multiprocessing import Process, current_process

def process_images_on_gpu(gpu_id, entries, image_base_path, output_path, model_id, prompt):
    # GPU 설정
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[GPU {gpu_id}] Process started with {len(entries)} entries.")

    # 모델과 프로세서 로드
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    processor = LlavaProcessor.from_pretrained(model_id)

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for entry in tqdm(entries, desc=f"[GPU {gpu_id}]"):
            image_path = os.path.join(image_base_path, entry["file_path"])
            try:
                raw_image = Image.open(image_path).convert("RGB")

                inputs = processor(
                    text=prompt,
                    images=raw_image,
                    return_tensors='pt'
                ).to(device, torch.float16)

                output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
                decoded_output = processor.decode(output[0][2:], skip_special_tokens=True).strip()
                entry["output"] = decoded_output
            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing {entry['file_path']}: {e}")
                entry["output"] = "ERROR"

            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[GPU {gpu_id}] Finished.")


if __name__ == "__main__":
    # 설정
    model_id = "bczhou/tiny-llava-v1-hf"
    json_input_path = "/home/aikusrv04/aiku/small_korean_vlm/data/korean_caption_bench/json/meta.json"
    image_base_path = "/home/aikusrv04/aiku/small_korean_vlm/data/korean_caption_bench/"
    output_base_path = "/home/aikusrv04/aiku/small_korean_vlm/data/korean_caption_bench/json/tinyllava_gpu"
    prompt = "USER: <image>\n이미지를 한 문장으로 자세히 한국어로 설명해줘\n"
    num_gpus = 4

    # JSON 데이터 로드
    with open(json_input_path, 'r') as f:
        json_data = json.load(f)

    # 데이터를 GPU 수만큼 분할
    chunk_size = (len(json_data) + num_gpus - 1) // num_gpus
    processes = []

    for i in range(num_gpus):
        chunk = json_data[i * chunk_size: (i + 1) * chunk_size]
        output_path = f"{output_base_path}{i}.jsonl"
        p = Process(
            target=process_images_on_gpu,
            args=(i, chunk, image_base_path, output_path, model_id, prompt)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All processes completed.")
