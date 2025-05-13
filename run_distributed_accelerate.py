import argparse
import json
import os

# import subprocess
import torch
import wandb
from datasets import load_dataset
from huggingface_hub.hf_api import HfFolder
from loguru import logger
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from logging_class import start_queue, write_log

# ---------------------------------------------------------------------------
HfFolder.save_token("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")
wandb.login("allow", "cd65e4ccbe4a97f6b8358f78f8ecf054f21466d9")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
parser = argparse.ArgumentParser(description="AIxBlock")
parser.add_argument(
    "--training_args_json",
    type=str,
    default=None,
    help="JSON string for training arguments",
)
parser.add_argument("--dataset_local", type=str, default=None, help="dataset id")
parser.add_argument("--channel_log", type=str, default=None, help="channel_log")
parser.add_argument("--hf_model_id", type=str, default=None, help="hf_model_id")
parser.add_argument("--push_to_hub", type=str, default=None, help="push_to_hub")
parser.add_argument(
    "--push_to_hub_token", type=str, default=None, help="push_to_hub_token"
)
parser.add_argument("--model_id", type=str, default=None, help="model_id")
parser.add_argument(
    "--dataset_id",
    type=str,
    default="autoprogrammer/Qwen2.5-Coder-7B-Instruct-codeguardplus",
    help="Name of the dataset to use",
)
parser.add_argument(
    "--prompt_field",
    type=str,
    default="prompt",
    help="Field name for prompts in the dataset",
)
parser.add_argument(
    "--text_field",
    type=str,
    default="response",
    help="Field name for text in the dataset",
)

args = parser.parse_args()
log_queue, logging_thread = start_queue(args.channel_log)
write_log(log_queue)
dataset_local = args.dataset_local
is_use_local = False
num_train_epochs = 1
per_train_dataset = 0.8
per_test_dataset = 0.2
output_dir = "./data/checkpoint"

# push_to_hub = True if args.push_to_hub and args.push_to_hub == "True" else False
push_to_hub = True
hf_model_id = args.hf_model_id if args.hf_model_id else "aixblock"
push_to_hub_token = "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"

if args.training_args_json:
    with open(args.training_args_json, "r") as f:
        training_args_dict = json.load(f)
else:
    training_args_dict = {}

if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
    compute_dtype = torch.float16
    attn_implementation = "sdpa"
torch.set_grad_enabled(True)
model_name = args.model_id if args.model_id else "google/gemma-3-4b-it"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(
    model_name, add_eos_token=True, use_fast=True, trust_remote_code=True
)
EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Prompt:
                {}

                ### Response:
                {}"""


def formatting_prompts_func(examples):
    prompt = examples.get(args.prompt_field)
    response = examples.get(args.text_field)
    texts = []
    for input, output in zip(prompt, response):
        text = alpaca_prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )


def tokenizer_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return tokenizer(
        "".join(texts),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )


if not training_args_dict:
    training_args_dict = {}

dataset_id = args.dataset_id
dataset_id = training_args_dict.get("dataset_id", dataset_id)


is_use_local = dataset_local is not None and dataset_local != "None"
if is_use_local:
    dataset_id = dataset_local


num_train_epochs = int(training_args_dict.get("num_train_epochs", 1))
per_train_dataset = float(training_args_dict.get("per_train_dataset", 0.8))
per_test_dataset = float(training_args_dict.get("per_test_dataset", 0.2))

# sfttrainer_args = {}


def formatting_func(example):
    text = f"{example['instruction']} {example['input']} {example['output']}"
    return {"text": text}


if not is_use_local:
    dataset = load_dataset(dataset_id)
    # Truy cập từng tập dữ liệu
    train_test_split = dataset["train"].train_test_split(
        test_size=per_test_dataset, seed=42
    )  # 20% cho eval
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    train_dataset = train_dataset.map(
        formatting_prompts_func, remove_columns=train_dataset.column_names, batched=True
    )
    eval_dataset = eval_dataset.map(
        formatting_prompts_func, remove_columns=eval_dataset.column_names, batched=True
    )
    sfttrainer_args = {
        "dataset_text_field": "text",
    }

else:
    # Load dataset từ thư mục local
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{dataset_id}/train_data.json",
            "test": f"{dataset_id}/test_data.json",
        },
    )

    # Truy cập từng tập dữ liệu
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    train_dataset = train_dataset.map(
        formatting_prompts_func, remove_columns=train_dataset.column_names, batched=True
    )
    eval_dataset = eval_dataset.map(
        formatting_prompts_func, remove_columns=eval_dataset.column_names, batched=True
    )
    sfttrainer_args = {
        "dataset_text_field": "text",
    }

# region Model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=compute_dtype,
)

model.enable_input_require_grads()

# Configure LoRA for Qwen model
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

training_arguments = TrainingArguments(
    output_dir="./data/checkpoint",
    eval_strategy="steps",
    do_eval=True,
    # optim="paged_adamw_8bit",
    per_device_train_batch_size=1,
    # gradient_accumulation_steps=4,
    per_device_eval_batch_size=1,
    log_level="debug",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=1e-4,
    # fp16 = not torch.cuda.is_bf16_supported(),
    # bf16 = torch.cuda.is_bf16_supported(),
    eval_steps=30,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    remove_unused_columns=False,
    report_to="tensorboard",  # azure_ml, comet_ml, mlflow, neptune, tensorboard, wandb, codecarbon, clearml, dagshub, flyte, dvclive
    push_to_hub=push_to_hub,
    push_to_hub_token=push_to_hub_token,
    no_cuda=False,
    push_to_hub_model_id=hf_model_id,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    # max_seq_length=512,
    # tokenizer=tokenizer,
    args=training_arguments,
    # dataset_kwargs={
    #                 "add_special_tokens": False,  # We template with special tokens
    #                 "append_concat_token": False, # No need to add additional separator token
    #                 'skip_prepare_dataset': True # skip the dataset preparation
    #             },
)

try:
    trainer.train()
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("⚠️ CUDA OOM detected, switching to CPU...")

        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()

        # Move model to CPU
        model.to("cpu")

        # Cập nhật lại args nếu cần (ví dụ bạn có logic chọn fp16 / bf16)
        training_arguments.fp16 = False
        training_arguments.bf16 = False
        training_arguments.no_cuda = True

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_arguments,
        )

        trainer.train()
    else:
        raise

trainer.push_to_hub()
output_dir = os.path.join("./data/checkpoint", hf_model_id.split("/")[-1])
trainer.save_model(output_dir)
# free the memory again
del model
del trainer
torch.cuda.empty_cache()
