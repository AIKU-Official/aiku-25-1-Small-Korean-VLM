import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor
import argparse
import re
import json
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="bczhou/tiny-llava-v1-hf")
parser.add_argument("--source_path", type=str, default=None)
parser.add_argument("--image_path", type=str, default=None)
parser.add_argument("--json_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--start_idx", type=int, default=0)        
args = parser.parse_args()




# model 세팅
model_id = args.model_name_or_path
image_folder = args.image_path
with open(args.json_path, 'r') as f:
    json_data = json.load(f)

model = LlavaForConditionalGeneration.from_pretrained(
          model_id,
          torch_dtype=torch.float16,
          low_cpu_mem_usage=True,
          ).to(0)
processor = LlavaProcessor.from_pretrained(model_id)




# inference
all_results = []
start = args.start_idx
pbar = tqdm(total=len(json_data[start:]), initial=start, desc="Overall progress")

for idx, entry in enumerate(json_data[start:], start=start):
    result = {
        "id" : entry['id'],
        "image": entry['image'],
    }
    
    image_name = os.path.basename(entry['image'])
    image_path = f"{image_folder}/{image_name}"
    try:
        raw_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        continue       
        
    new_conversations = []
    for index, queries in enumerate(entry['conversations']):
        # breakpoint()
        if index%4==0 or index%4==2:
            question = queries['value']
            answer = entry['conversations'][index+1]['value']

            if "<image>" not in question:
                    prompt = f"<image>\n{question}"
            else:
                    prompt = question
                    
            new_conversations.append({
                    "from": "human",
                    "value": prompt
                })
            new_conversations.append({
                    "from": "gpt",
                    "value": answer
                })
                
                
            inputs = processor(text=prompt, images=raw_image, return_tensors='pt', padding=True).to(0, torch.float16)
            # breakpoint()
            output = model.generate(

                            **inputs,
                            max_new_tokens=100,
                            do_sample=False
                        )
            decoded_output = processor.decode(output[0][2:], skip_special_tokens=True)

            new_conversations.append({
                    "from": "pred",
                    "value": decoded_output.strip()
                })
            print(f"질문 : {question}")
            print(f"답변 : {decoded_output.strip()}")
            
    result['conversations'] = new_conversations
    all_results.append(result)
    pbar.update(1)

pbar.close()

# 저장
output_folder = args.output_path
filename = "tinyllava_chartqa.json"
with open(f"{output_folder}/{filename}", 'w') as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)
print(f"Results saved to {output_folder}/{filename}")
