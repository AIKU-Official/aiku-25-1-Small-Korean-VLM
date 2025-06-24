from huggingface_hub import HfApi

token = "YOUR_HUGGINGFACE_TOKEN"  # Replace with your actual Hugging Face token

api = HfApi(token=token)
api.upload_folder(
    folder_path="/home/aikusrv04/aiku/small_korean_vlm/checkpoints/lora_merged/svlm_data_augmented",
    repo_id="iamnotwhale/llava-svlm",
    repo_type="model",
)
