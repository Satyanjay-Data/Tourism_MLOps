import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

folder_path = "tourism_project/deployment"  # ✅ FIXED PATH

api.upload_folder(
    folder_path=folder_path,
    repo_id="Satyanjay/tourism-package-prediction-model",
    repo_type="model",
)
