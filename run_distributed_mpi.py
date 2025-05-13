import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer

# from safetensors.torch import load_model

parser = argparse.ArgumentParser(description="AIxBlock")
parser.add_argument(
    "--backend", type=str, default="nccl", help="JSON string for backend"
)
parser.add_argument("--hf_model_id", type=str, default=None, help="hf_model_id")
parser.add_argument("--world_size", type=int, default=1, help="world_size")
parser.add_argument("--master_addr", type=str, default=1, help="master_addr")
parser.add_argument("--master_port", type=str, default=1, help="master_port")
parser.add_argument("--rank", type=str, default=1, help="rank")

parser.add_argument("--prompt", type=str, default=None, help="dataset id")
parser.add_argument("--text", type=str, default=None, help="channel_log")
parser.add_argument("--hf_token", type=str, default=None, help="hf_token")
parser.add_argument("--max_new_tokens", type=int, default=1024, help="world_size")

# Phân tích các tham số dòng lệnh
args = parser.parse_args()


# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse

import evaluate
import torch
from accelerate import Accelerator, DistributedType
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup, set_seed)

########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a Bert base model on GLUE MRPC
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################


MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32


def get_dataloaders(accelerator: Accelerator, batch_size: int = 16):
    """
    Creates a set of `DataLoader`s for the `glue` dataset,
    using "bert-base-cased" as the tokenizer.

    Args:
        accelerator (`Accelerator`):
            An `Accelerator` object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    datasets = load_dataset("glue", "mrpc")

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            max_length=None,
        )
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence1", "sentence2"],
        )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        # For Torchxla, it's best to pad everything to the same length or training will be very slow.
        max_length = (
            128 if accelerator.distributed_type == DistributedType.XLA else None
        )
        # When using mixed precision we want round multiples of 8/16
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        return tokenizer.pad(
            examples,
            padding="longest",
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=EVAL_BATCH_SIZE,
        drop_last=(accelerator.mixed_precision == "fp8"),
    )

    return train_dataloader, eval_dataloader


def training_function(config, args):
    # Initialize accelerator
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])

    metric = evaluate.load("glue", "mrpc")

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if (
        batch_size > MAX_GPU_BATCH_SIZE
        and accelerator.distributed_type != DistributedType.XLA
    ):
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    set_seed(seed)
    train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", return_dict=True
    )

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)
    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * num_epochs)
        // gradient_accumulation_steps,
    )

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    )

    # Now we train the model
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics(
                (predictions, batch["labels"])
            )
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}:", eval_metric)
    accelerator.end_training()


parser = argparse.ArgumentParser(description="Simple example of training script.")
parser.add_argument(
    "--mixed_precision",
    type=str,
    default=None,
    choices=["no", "fp16", "bf16", "fp8"],
    help="Whether to use mixed precision. Choose"
    "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
    "and an Nvidia Ampere GPU.",
)
parser.add_argument(
    "--cpu", action="store_true", help="If passed, will train on the CPU."
)
args = parser.parse_args()
config = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 16}
training_function(config, args)


# export CCL_WORKER_COUNT=1
# export MASTER_ADDR=xxx.xxx.xxx.xxx #node0 ip
# mpirun -f hostfile -n 4 -ppn 2 \
# -genv OMP_NUM_THREADS=23 \
# python3 run_qa.py \
#     --model_name_or_path distilbert-base-uncased-distilled-squad \
#     --dataset_name squad \
#     --apply_quantization \
#     --quantization_approach static \
#     --do_train \
#     --do_eval \
#     --verify_loading \
#     --output_dir /tmp/squad_output \
#     --no_cuda \
#     --xpu_backend ccl


# export CCL_WORKER_COUNT=1
# export MASTER_ADDR=xxx.xxx.xxx.xxx #node0 ip
# export CCL_ATL_TRANSPORT=ofi
# mpirun -f hostfile -n 16 -ppn 4 accelerate launch run_distributed_mpi.py
# https://github.com/huggingface/accelerate/blob/main/docs/source/usage_guides/ipex.md#how-it-works-for-training-optimization-in-cpu
# https://github.com/Qualcomm-AI-research/outlier-free-transformers/blob/main/run_mlm.py
