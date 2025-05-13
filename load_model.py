import os

import torch
from huggingface_hub import HfFolder
from transformers import AutoProcessor, SeamlessM4TModel

# -----------------------------------------------------------------------
hf_token = os.getenv("HF_TOKEN", "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")
HfFolder.save_token(hf_token)

from huggingface_hub import login

login(token=hf_token)


def load():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_id = "facebook/hf-seamless-m4t-medium"
    processor = AutoProcessor.from_pretrained(model_id)
    model = SeamlessM4TModel.from_pretrained(model_id).to(device)


load()
