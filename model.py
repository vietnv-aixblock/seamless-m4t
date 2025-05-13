# model_marketplace.config
# {
#     "token_length": "128000",
#     "sampling_frequency": "44100",
#     "framework": "transformers",
#     "dataset_format": "llm",
#     "dataset_sample": "[id on s3]",
#     "weights": [
#         {
#             "name": "gemma-3-27b-it",
#             "value": "google/gemma-3-27b-it",
#             "size": 140,
#             "parameters": "27b",
#             "tflops": 70,
#             "vram": 82,
#             "nodes": 2,
#         },
#         {
#             "name": "gemma-3-12b-it",
#             "value": "google/gemma-3-12b-it",
#             "size": 65,
#             "parameters": "12b",
#             "tflops": 32,
#             "vram": 32,
#             "nodes": 2,
#         },
#         {
#             "name": "gemma-3-4b-it",
#             "value": "google/gemma-3-4b-it",
#             "size": 25,
#             "parameters": "4b",
#             "tflops": 11,
#             "vram": 11,
#             "nodes": 1,
#         },
#         {
#             "name": "gemma-3-1b-it",
#             "value": "google/gemma-3-1b-it",
#             "size": 5,
#             "parameters": "1b",
#             "tflops": 4,
#             "vram": 3,
#             "nodes": 1,
#         },
#     ],
#     "cuda": "12.6",
#     "task": [
#         "text-generation",
#         "question-answering",
#         "image-text-to-text",
#     ],
# }
import base64
import io
import os
import subprocess
import sys
import threading
from typing import Iterator

import gradio as gr
import torch
from aixblock_ml.model import AIxBlockMLBase
from huggingface_hub import HfFolder, login
from loguru import logger
from mcp.server.fastmcp import FastMCP

from gradio_app import *
import gc

# ------------------------------------------------------------------------------
hf_token = os.getenv("HF_TOKEN", "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")
HfFolder.save_token(hf_token)


hf_access_token = "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"
login(token=hf_access_token)
CUDA_VISIBLE_DEVICES = []
for i in range(torch.cuda.device_count()):
    CUDA_VISIBLE_DEVICES.append(i)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    f"{i}" for i in range(len(CUDA_VISIBLE_DEVICES))
)
print(os.environ["CUDA_VISIBLE_DEVICES"])


HOST_NAME = os.environ.get("HOST_NAME", "https://dev-us-west-1.aixblock.io")
TYPE_ENV = os.environ.get("TYPE_ENV", "DETECTION")


mcp = FastMCP("aixblock-mcp")

CHANNEL_STATUS = {}

model_demo = None
processor_demo = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MyModel(AIxBlockMLBase):

    @mcp.tool()
    def action(self, command, **kwargs):
        print(
            f"""
                command: {command}
            """
        )
        if command.lower() == "execute":
            _command = kwargs.get("shell", None)
            logger.info(f"Executing command: {_command}")
            subprocess.Popen(
                _command,
                shell=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
            )
            return {"message": "command completed successfully"}

        elif command.lower() == "train":

            # model_id = kwargs.get(
            #     "model_id", "google/gemma-3-4b-it"
            # )
            # dataset_id = kwargs.get(
            #     "dataset_id", "autoprogrammer/Qwen2.5-Coder-7B-Instruct-codeguardplus"
            # )

            # push_to_hub = kwargs.get("push_to_hub", True)
            # hf_model_id = kwargs.get(
            #     "hf_model_id", "google/gemma-3-4b-it"
            # )
            # push_to_hub_token = kwargs.get(
            #     "push_to_hub_token", "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"
            # )
            # framework = kwargs.get("framework", "huggingface")
            # task = kwargs.get("task", "text-generation")
            # prompt = kwargs.get("prompt", "")
            # trainingArguments = kwargs.get("TrainingArguments", None)
            # cuda_debug = kwargs.get("cuda_debug", False)

            # json_file = "training_args.json"
            # absolute_path = os.path.abspath(json_file)

            # with open(absolute_path, "w") as f:
            #     json.dump(trainingArguments, f)
            # print(trainingArguments)

            # if cuda_debug == True:
            #     os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
            #     os.environ["NCCL_DEBUG"] = "INFO"

            # os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
            # os.environ["TORCH_USE_CUDA_DSA"] = "0"
            # clone_dir = os.path.join(os.getcwd())
            # project_id = kwargs.get("project_id", 0)
            # token = kwargs.get("token", "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")
            # checkpoint_version = kwargs.get("checkpoint_version")
            # checkpoint_id = kwargs.get("checkpoint")
            # dataset_version = kwargs.get("dataset_version")
            # dataset_id = kwargs.get("dataset")
            # channel_log = kwargs.get("channel_log", "training_logs")
            # world_size = kwargs.get("world_size", 1)
            # rank = kwargs.get("rank", 0)
            # master_add = kwargs.get("master_add", "127.0.0.1")
            # master_port = kwargs.get("master_port", "23456")
            # host_name = kwargs.get("host_name", HOST_NAME)

            # log_queue, logging_thread = start_queue(channel_log)
            # write_log(log_queue)

            # channel_name = f"{hf_model_id}_{str(uuid.uuid4())[:8]}"
            # username = ""
            # hf_model_name = ""
            # CHANNEL_STATUS[channel_name] = {
            #     "status": "training",
            #     "hf_model_id": hf_model_name,
            #     "command": command,
            #     "created_at": time.time(),
            # }
            # print(f"🚀 Đã bắt đầu training kênh: {channel_name}")

            # def func_train_model(
            #     clone_dir,
            #     project_id,
            #     token,
            #     checkpoint_version,
            #     checkpoint_id,
            #     dataset_version,
            #     dataset_id,
            #     model_id,
            #     world_size,
            #     rank,
            #     master_add,
            #     master_port,
            #     prompt,
            #     json_file,
            #     channel_log,
            #     hf_model_id,
            #     push_to_hub,
            #     push_to_hub_token,
            #     host_name,
            # ):

            #     dataset_path = None
            #     project = connect_project(host_name, token, project_id)

            #     if dataset_version and dataset_id and project:
            #         dataset_path = os.path.join(
            #             clone_dir, f"datasets/{dataset_version}"
            #         )

            #         if not os.path.exists(dataset_path):
            #             data_path = os.path.join(clone_dir, "data_zip")
            #             os.makedirs(data_path, exist_ok=True)

            #             dataset_name = download_dataset(project, dataset_id, data_path)
            #             print(dataset_name)
            #             if dataset_name:
            #                 data_zip_dir = os.path.join(data_path, dataset_name)

            #                 with zipfile.ZipFile(data_zip_dir, "r") as zip_ref:
            #                     zip_ref.extractall(dataset_path)

            #                 extracted_files = os.listdir(dataset_path)
            #                 zip_files = [
            #                     f for f in extracted_files if f.endswith(".zip")
            #                 ]

            #                 if len(zip_files) == 1:
            #                     inner_zip_path = os.path.join(
            #                         dataset_path, zip_files[0]
            #                     )
            #                     print(
            #                         f"🔁 Found inner zip file: {inner_zip_path}, extracting..."
            #                     )
            #                     with zipfile.ZipFile(inner_zip_path, "r") as inner_zip:
            #                         inner_zip.extractall(dataset_path)
            #                     os.remove(inner_zip_path)

            #     subprocess.run(
            #         ("whereis accelerate"),
            #         shell=True,
            #     )
            #     print("===Train===")
            #     if framework == "huggingface":
            #         if int(world_size) > 1:
            #             if int(rank) == 0:
            #                 print("master node")
            #                 command = (
            #                     "venv/bin/accelerate launch --num_processes {num_processes} --num_machines {SLURM_NNODES} --machine_rank 0 --main_process_ip {head_node_ip} --main_process_port {port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id}"
            #                 ).format(
            #                     num_processes=world_size * torch.cuda.device_count(),
            #                     SLURM_NNODES=world_size,
            #                     head_node_ip=master_add,
            #                     port=master_port,
            #                     file_name="./run_distributed_accelerate.py",
            #                     json_file=json_file,
            #                     dataset_path=dataset_path,
            #                     channel_log=channel_log,
            #                     hf_model_id=hf_model_id,
            #                     push_to_hub=push_to_hub,
            #                     model_id=model_id,
            #                     push_to_hub_token=push_to_hub_token,
            #                 )
            #                 process = subprocess.run(
            #                     command,
            #                     shell=True,
            #                 )
            #             else:
            #                 print("worker node")
            #                 command = (
            #                     "venv/bin/accelerate launch --num_processes {num_processes} --num_machines {SLURM_NNODES} --machine_rank {machine_rank} --main_process_ip {head_node_ip} --main_process_port {port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id}"
            #                 ).format(
            #                     num_processes=world_size * torch.cuda.device_count(),
            #                     SLURM_NNODES=world_size,
            #                     head_node_ip=master_add,
            #                     port=master_port,
            #                     machine_rank=rank,
            #                     file_name="./run_distributed_accelerate.py",
            #                     json_file=json_file,
            #                     dataset_path=dataset_path,
            #                     channel_log=channel_log,
            #                     hf_model_id=hf_model_id,
            #                     push_to_hub=push_to_hub,
            #                     model_id=model_id,
            #                     push_to_hub_token=push_to_hub_token,
            #                 )
            #                 process = subprocess.run(
            #                     command,
            #                     shell=True,
            #                 )

            #         else:
            #             if torch.cuda.device_count() > 1:  # multi gpu
            #                 command = (
            #                     "venv/bin/accelerate launch --multi_gpu --num_machines {SLURM_NNODES} --machine_rank 0 --num_processes {num_processes} {file_name} --training_args_json {json_file}  --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id}"
            #                 ).format(
            #                     num_processes=world_size * torch.cuda.device_count(),
            #                     SLURM_NNODES=world_size,
            #                     # head_node_ip=os.environ.get("head_node_ip", master_add),
            #                     port=master_port,
            #                     file_name="./run_distributed_accelerate.py",
            #                     json_file=json_file,
            #                     dataset_path=dataset_path,
            #                     channel_log=channel_log,
            #                     hf_model_id=hf_model_id,
            #                     push_to_hub=push_to_hub,
            #                     model_id=model_id,
            #                     push_to_hub_token=push_to_hub_token,
            #                 )
            #                 print("================2")
            #                 print(command)
            #                 print("================2")
            #                 process = subprocess.run(command, shell=True)

            #             elif torch.cuda.device_count() == 1:  # one gpu
            #                 command = (
            #                     "venv/bin/accelerate launch {file_name} --training_args_json {json_file}  --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id}"
            #                 ).format(
            #                     file_name="./run_distributed_accelerate.py",
            #                     json_file=json_file,
            #                     dataset_path=dataset_path,
            #                     channel_log=channel_log,
            #                     hf_model_id=hf_model_id,
            #                     push_to_hub={push_to_hub},
            #                     model_id=model_id,
            #                     push_to_hub_token={push_to_hub_token},
            #                 )
            #                 print("================")
            #                 print(command)
            #                 print("================")
            #                 process = subprocess.Popen(
            #                     command,
            #                     shell=True,
            #                     stdout=sys.stdout,
            #                     stderr=sys.stderr,
            #                     text=True,
            #                 )
            #             else:  # no gpu
            #                 command = (
            #                     "venv/bin/accelerate launch --cpu {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id}"
            #                 ).format(
            #                     file_name="./run_distributed_accelerate.py",
            #                     json_file=json_file,
            #                     dataset_path=dataset_path,
            #                     channel_log=channel_log,
            #                     hf_model_id=hf_model_id,
            #                     push_to_hub=push_to_hub,
            #                     model_id=model_id,
            #                     push_to_hub_token=push_to_hub_token,
            #                 )

            #                 process = subprocess.Popen(
            #                     command,
            #                     shell=True,
            #                     stdout=sys.stdout,
            #                     stderr=sys.stderr,
            #                     text=True,
            #                 )
            #                 while True:
            #                     output = process.stdout.readline().decode("utf-8")
            #                     if output == "" and process.poll() is not None:
            #                         break
            #                     if output:
            #                         print(output, end="")
            #                 process.wait()

            #     elif framework == "pytorch":
            #         process = subprocess.run(
            #             ("whereis torchrun"),
            #             shell=True,
            #         )

            #         if int(world_size) > 1:
            #             if rank == 0:
            #                 print("master node")
            #                 command = (
            #                     "venv/bin/torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
            #                     "--master_addr {master_addr} --master_port {master_port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id}"
            #                 ).format(
            #                     nnodes=int(world_size),
            #                     node_rank=int(rank),
            #                     nproc_per_node=world_size * torch.cuda.device_count(),
            #                     master_addr="127.0.0.1",
            #                     master_port="23456",
            #                     file_name="./run_distributed_accelerate.py",
            #                     json_file=json_file,
            #                     dataset_path=dataset_path,
            #                     channel_log=channel_log,
            #                     hf_model_id=hf_model_id,
            #                     push_to_hub=push_to_hub,
            #                     model_id=model_id,
            #                     push_to_hub_token=push_to_hub_token,
            #                 )
            #                 process = subprocess.Popen(
            #                     command,
            #                     shell=True,
            #                     stdout=sys.stdout,
            #                     stderr=sys.stderr,
            #                     text=True,
            #                 )
            #             else:
            #                 print("worker node")
            #                 command = (
            #                     "venv/bin/torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
            #                     "--master_addr {master_addr} --master_port {master_port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id}"
            #                 ).format(
            #                     nnodes=int(world_size),
            #                     node_rank=int(rank),
            #                     nproc_per_node=world_size * torch.cuda.device_count(),
            #                     master_addr=master_add,
            #                     master_port=master_port,
            #                     file_name="./run_distributed_accelerate.py",
            #                     json_file=json_file,
            #                     dataset_path=dataset_path,
            #                     channel_log=channel_log,
            #                     hf_model_id=hf_model_id,
            #                     push_to_hub=push_to_hub,
            #                     model_id=model_id,
            #                     push_to_hub_token=push_to_hub_token,
            #                 )
            #                 print(command)
            #                 process = subprocess.Popen(
            #                     command,
            #                     shell=True,
            #                     stdout=sys.stdout,
            #                     stderr=sys.stderr,
            #                     text=True,
            #                 )
            #         else:
            #             command = (
            #                 "venv/bin/torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
            #                 "{file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id}"
            #             ).format(
            #                 nnodes=int(world_size),
            #                 node_rank=int(rank),
            #                 nproc_per_node=world_size * torch.cuda.device_count(),
            #                 file_name="./run_distributed_accelerate.py",
            #                 json_file=json_file,
            #                 dataset_path=dataset_path,
            #                 channel_log=channel_log,
            #                 hf_model_id=hf_model_id,
            #                 push_to_hub=push_to_hub,
            #                 model_id=model_id,
            #                 push_to_hub_token=push_to_hub_token,
            #             )
            #             process = subprocess.run(
            #                 command,
            #                 shell=True,
            #             )
            #     CHANNEL_STATUS[channel_name]["status"] = "done"
            #     output_dir = "./data/checkpoint"
            #     print(push_to_hub)
            #     if push_to_hub:
            #         import datetime

            #         output_dir = "./data/checkpoint"
            #         now = datetime.datetime.now()
            #         date_str = now.strftime("%Y%m%d")
            #         time_str = now.strftime("%H%M%S")
            #         version = f"{date_str}-{time_str}"

            #         upload_checkpoint(project, version, output_dir)

            # train_thread = threading.Thread(
            #     target=func_train_model,
            #     args=(
            #         clone_dir,
            #         project_id,
            #         token,
            #         checkpoint_version,
            #         checkpoint_id,
            #         dataset_version,
            #         dataset_id,
            #         model_id,
            #         world_size,
            #         rank,
            #         master_add,
            #         master_port,
            #         prompt,
            #         absolute_path,
            #         channel_log,
            #         hf_model_id,
            #         push_to_hub,
            #         push_to_hub_token,
            #         host_name,
            #     ),
            # )
            # train_thread.start()

            return {
                "message": "train completed successfully",
                # "channel_name": channel_name,
            }
        elif command.lower() == "stop":
            subprocess.run(["pkill", "-9", "-f", "./inference/generate.py"])
            return {"message": "train stop successfully", "result": "Done"}

        elif command.lower() == "tensorboard":

            def run_tensorboard():
                p = subprocess.Popen(
                    f"tensorboard --logdir /app/data/checkpoint/runs --host 0.0.0.0 --port=6006",
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    text=True,
                )
                out = p.communicate()
                print(out)

            tensorboard_thread = threading.Thread(target=run_tensorboard)
            tensorboard_thread.start()
            return {"message": "tensorboardx started successfully"}
        # region Predict
        elif command.lower() == "predict":
            model_id = kwargs.get("model_id", "facebook/hf-seamless-m4t-medium")
            text = kwargs.get("text", None)
            task = kwargs.get("task", None)
            voice = kwargs.get("voice", None)
            target_language = kwargs.get("target_language", "English")
            # Process Audio
            audio_bytes = base64.b64decode(voice)
            audio_buffer = io.BytesIO(audio_bytes)
            with torch.no_grad():
                processor = AutoProcessor.from_pretrained(model_id)
                model = SeamlessM4TModel.from_pretrained(model_id).to(device)
                audio_out, text_out = predict(
                    task_name=task,
                    audio_source="file",
                    input_audio_file=audio_buffer,
                    input_text=text,
                    source_language=None,
                    target_language=target_language,
                    model_demo=model,
                    processor_demo=processor,
                    device=device,
                )
            if audio_out:
                rate, waveform = audio_out
            else:
                waveform = None

            del processor, model
            gc.collect()
            torch.cuda.empty_cache()
            return {
                "message": "predict completed successfully",
                "result": {"audio_out": waveform, "text_out": text_out},
            }
        elif command.lower() == "prompt_sample":
            task = kwargs.get("task", "")
            if task == "question-answering":
                prompt_text = f"""
                    Here is the context: 
                    {{context}}

                    Based on the above context, provide an answer to the following question: 
                    {{question}}

                    Answer:
                    """
            elif task == "text-classification":
                prompt_text = f"""
                    Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

                    Text: 
                    {{context}}

                    Summary:
                    """

            elif task == "summarization":
                prompt_text = f"""
                    Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

                    Text: 
                    {{context}}

                    Summary:
                    """
            return {
                "message": "prompt_sample completed successfully",
                "result": prompt_text,
            }

        elif command.lower() == "action-example":
            return {"message": "Done", "result": "Done"}

        elif command == "status":
            channel = kwargs.get("channel", None)

            if channel:
                # Nếu có truyền kênh cụ thể
                status_info = CHANNEL_STATUS.get(channel)
                if status_info is None:
                    return {"channel": channel, "status": "not_found"}
                elif isinstance(status_info, dict):
                    return {"channel": channel, **status_info}
                else:
                    return {"channel": channel, "status": status_info}
            else:
                # Lấy tất cả kênh
                if not CHANNEL_STATUS:
                    return {"message": "No channels available"}

                channels = []
                for ch, info in CHANNEL_STATUS.items():
                    if isinstance(info, dict):
                        channels.append({"channel": ch, **info})
                    else:
                        channels.append({"channel": ch, "status": info})

                return {"channels": channels}
        else:
            return {"message": "command not supported", "result": None}

    @mcp.tool()
    def model(self, **kwargs):
        # -----------------------------------------------------------------------------------------------------------------
        DESCRIPTION = """# SeamlessM4T
        [SeamlessM4T](https://github.com/facebookresearch/seamless_communication) is designed to provide high-quality
        translation, allowing people from different linguistic communities to communicate effortlessly through speech and text.
        This unified model enables multiple tasks like Speech-to-Speech (S2ST), Speech-to-Text (S2TT), Text-to-Speech (T2ST)
        translation and more, without relying on multiple separate models.
        """

        CACHE_EXAMPLES = os.getenv("CACHE_EXAMPLES") == "1"

        TASK_NAMES = [
            "S2ST (Speech to Speech translation)",
            "S2TT (Speech to Text translation)",
            "T2ST (Text to Speech translation)",
            "T2TT (Text to Text translation)",
            "ASR (Automatic Speech Recognition)",
        ]
        AUDIO_SAMPLE_RATE = 16000.0
        MAX_INPUT_AUDIO_LENGTH = 60  # in seconds
        DEFAULT_TARGET_LANGUAGE = "French"

        def process_s2st_example(
            input_audio_file: str, target_language: str
        ) -> tuple[tuple[int, np.ndarray] | None, str]:
            return _predict(
                task_name="S2ST",
                audio_source="file",
                input_audio_mic=None,
                input_audio_file=input_audio_file,
                input_text=None,
                source_language=None,
                target_language=target_language,
            )

        def process_s2tt_example(
            input_audio_file: str, target_language: str
        ) -> tuple[tuple[int, np.ndarray] | None, str]:
            return _predict(
                task_name="S2TT",
                audio_source="file",
                input_audio_mic=None,
                input_audio_file=input_audio_file,
                input_text=None,
                source_language=None,
                target_language=target_language,
            )

        def process_t2st_example(
            input_text: str, source_language: str, target_language: str
        ) -> tuple[tuple[int, np.ndarray] | None, str]:
            return _predict(
                task_name="T2ST",
                audio_source="",
                input_audio_mic=None,
                input_audio_file=None,
                input_text=input_text,
                source_language=source_language,
                target_language=target_language,
            )

        def process_t2tt_example(
            input_text: str, source_language: str, target_language: str
        ) -> tuple[tuple[int, np.ndarray] | None, str]:
            return _predict(
                task_name="T2TT",
                audio_source="",
                input_audio_mic=None,
                input_audio_file=None,
                input_text=input_text,
                source_language=source_language,
                target_language=target_language,
            )

        def process_asr_example(
            input_audio_file: str, target_language: str
        ) -> tuple[tuple[int, np.ndarray] | None, str]:
            return _predict(
                task_name="ASR",
                audio_source="file",
                input_audio_mic=None,
                input_audio_file=input_audio_file,
                input_text=None,
                source_language=None,
                target_language=target_language,
            )

        def update_audio_ui(audio_source: str) -> tuple[dict, dict]:
            mic = audio_source == "microphone"
            return (
                gr.update(visible=mic, value=None),  # input_audio_mic
                gr.update(visible=not mic, value=None),  # input_audio_file
            )

        def update_input_ui(task_name: str) -> tuple[dict, dict, dict, dict]:
            task_name = task_name.split()[0]
            if task_name == "S2ST":
                return (
                    gr.update(visible=True),  # audio_box
                    gr.update(visible=False),  # input_text
                    gr.update(visible=False),  # source_language
                    gr.update(
                        visible=True,
                        choices=S2ST_TARGET_LANGUAGE_NAMES,
                        value=DEFAULT_TARGET_LANGUAGE,
                    ),  # target_language
                )
            elif task_name == "S2TT":
                return (
                    gr.update(visible=True),  # audio_box
                    gr.update(visible=False),  # input_text
                    gr.update(visible=False),  # source_language
                    gr.update(
                        visible=True,
                        choices=S2TT_TARGET_LANGUAGE_NAMES,
                        value=DEFAULT_TARGET_LANGUAGE,
                    ),  # target_language
                )
            elif task_name == "T2ST":
                return (
                    gr.update(visible=False),  # audio_box
                    gr.update(visible=True),  # input_text
                    gr.update(visible=True),  # source_language
                    gr.update(
                        visible=True,
                        choices=S2ST_TARGET_LANGUAGE_NAMES,
                        value=DEFAULT_TARGET_LANGUAGE,
                    ),  # target_language
                )
            elif task_name == "T2TT":
                return (
                    gr.update(visible=False),  # audio_box
                    gr.update(visible=True),  # input_text
                    gr.update(visible=True),  # source_language
                    gr.update(
                        visible=True,
                        choices=T2TT_TARGET_LANGUAGE_NAMES,
                        value=DEFAULT_TARGET_LANGUAGE,
                    ),  # target_language
                )
            elif task_name == "ASR":
                return (
                    gr.update(visible=True),  # audio_box
                    gr.update(visible=False),  # input_text
                    gr.update(visible=False),  # source_language
                    gr.update(
                        visible=True,
                        choices=S2TT_TARGET_LANGUAGE_NAMES,
                        value=DEFAULT_TARGET_LANGUAGE,
                    ),  # target_language
                )
            else:
                raise ValueError(f"Unknown task: {task_name}")

        def update_output_ui(task_name: str) -> tuple[dict, dict]:
            task_name = task_name.split()[0]
            if task_name in ["S2ST", "T2ST"]:
                return (
                    gr.update(visible=True, value=None),  # output_audio
                    gr.update(value=None),  # output_text
                )
            elif task_name in ["S2TT", "T2TT", "ASR"]:
                return (
                    gr.update(visible=False, value=None),  # output_audio
                    gr.update(value=None),  # output_text
                )
            else:
                raise ValueError(f"Unknown task: {task_name}")

        def update_example_ui(task_name: str) -> tuple[dict, dict, dict, dict, dict]:
            task_name = task_name.split()[0]
            return (
                gr.update(visible=task_name == "S2ST"),  # s2st_example_row
                gr.update(visible=task_name == "S2TT"),  # s2tt_example_row
                gr.update(visible=task_name == "T2ST"),  # t2st_example_row
                gr.update(visible=task_name == "T2TT"),  # t2tt_example_row
                gr.update(visible=task_name == "ASR"),  # asr_example_row
            )

        def _load_model():
            global model_demo, processor_demo
            model_id = "facebook/hf-seamless-m4t-medium"
            processor_demo = AutoProcessor.from_pretrained(model_id)
            model_demo = SeamlessM4TModel.from_pretrained(model_id).to(device)
            return model_demo is not None

        def _predict(
            task_name: str,
            audio_source: str,
            input_audio_mic: str | None,
            input_audio_file: str | None,
            input_text: str | None,
            source_language: str | None,
            target_language: str,
            model_loaded: bool = False,
        ) -> tuple[tuple[int, np.ndarray] | None, str]:
            if model_demo == None:
                gr.Warning("Please load the model before translating.")
                return None, "Please load the model before translating."

            task_name = task_name.split()[0]
            source_language_code = (
                LANGUAGE_NAME_TO_CODE[source_language] if source_language else None
            )
            target_language_code = LANGUAGE_NAME_TO_CODE[target_language]

            if task_name in ["S2ST", "S2TT", "ASR"]:
                if audio_source == "microphone":
                    input_data = input_audio_mic
                else:
                    input_data = input_audio_file

                arr, org_sr = torchaudio.load(input_data)
                new_arr = torchaudio.functional.resample(
                    arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE
                )
                max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
                if new_arr.shape[1] > max_length:
                    new_arr = new_arr[:, :max_length]
                    gr.Warning(
                        f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used."
                    )

                input_data = processor_demo(
                    audios=new_arr, sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt"
                ).to(device)
            else:
                input_data = processor_demo(
                    text=input_text, src_lang=source_language_code, return_tensors="pt"
                ).to(device)

            if task_name in ["S2TT", "T2TT"]:
                tokens_ids = (
                    model_demo.generate(
                        **input_data,
                        generate_speech=False,
                        tgt_lang=target_language_code,
                        num_beams=5,
                        do_sample=True,
                    )[0]
                    .cpu()
                    .squeeze()
                    .detach()
                    .tolist()
                )
            else:
                output = model_demo.generate(
                    **input_data,
                    return_intermediate_token_ids=True,
                    tgt_lang=target_language_code,
                    num_beams=5,
                    do_sample=True,
                    spkr_id=LANG_TO_SPKR_ID[target_language_code][0],
                )
                waveform = output.waveform.cpu().squeeze().detach().numpy()
                tokens_ids = output.sequences.cpu().squeeze().detach().tolist()

            text_out = processor_demo.decode(tokens_ids, skip_special_tokens=True)

            if task_name in ["S2ST", "T2ST"]:
                return (AUDIO_SAMPLE_RATE, waveform), text_out
            else:
                return None, text_out

        with gr.Blocks(css="style.css") as demo:
            gr.Markdown(DESCRIPTION)
            gr.DuplicateButton(
                value="Duplicate Space for private use",
                elem_id="duplicate-button",
                visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
            )
            with gr.Row():
                btn_load_model = gr.Button("Load Model")
                model_loaded = gr.State(model_demo is not None)
                loading_msg = gr.Markdown("", visible=False)

            def check_model_loaded():
                return gr.update(visible=(model_demo is None))

            demo.load(fn=check_model_loaded, inputs=None, outputs=btn_load_model)


            def show_loading_and_disable_btn():
                # Hiện thông báo và disable nút
                return gr.update(
                    value="⏳ Đang tải mô hình...", visible=True
                ), gr.update(interactive=False)

            def hide_loading_and_hide_btn():
                # Ẩn thông báo và ẩn luôn nút
                return gr.update(value="", visible=False), gr.update(visible=False)

            btn_load_model.click(
                fn=show_loading_and_disable_btn,
                inputs=[],
                outputs=[loading_msg, btn_load_model],
                queue=False,
            ).then(fn=_load_model, inputs=[], outputs=[model_loaded], queue=True).then(
                fn=hide_loading_and_hide_btn,
                inputs=[],
                outputs=[loading_msg, btn_load_model],
                queue=False,
            )

            def toggle_btn_load_model(model_loaded):
                # Nếu model đã load thì ẩn nút, ngược lại thì hiện
                return gr.update(visible=not model_loaded)

            model_loaded.change(
                fn=toggle_btn_load_model,
                inputs=model_loaded,
                outputs=btn_load_model,
                queue=False,
            )

            with gr.Group():
                task_name = gr.Dropdown(
                    label="Task",
                    choices=TASK_NAMES,
                    value=TASK_NAMES[0],
                )
                with gr.Row():
                    source_language = gr.Dropdown(
                        label="Source language",
                        choices=TEXT_SOURCE_LANGUAGE_NAMES,
                        value="English",
                        visible=False,
                    )
                    target_language = gr.Dropdown(
                        label="Target language",
                        choices=S2ST_TARGET_LANGUAGE_NAMES,
                        value=DEFAULT_TARGET_LANGUAGE,
                    )
                with gr.Row() as audio_box:
                    audio_source = gr.Radio(
                        label="Audio source",
                        choices=["file", "microphone"],
                        value="file",
                    )
                    input_audio_mic = gr.Audio(
                        label="Input speech",
                        type="filepath",
                        sources="microphone",
                        visible=False,
                    )
                    input_audio_file = gr.Audio(
                        label="Input speech",
                        type="filepath",
                        sources="upload",
                        visible=True,
                    )
                input_text = gr.Textbox(label="Input text", visible=False)
                model_loaded = gr.State(False)
                btn = gr.Button("Translate")
                with gr.Column():
                    output_audio = gr.Audio(
                        label="Translated speech",
                        autoplay=False,
                        streaming=False,
                        type="numpy",
                    )
                    output_text = gr.Textbox(label="Translated text")

            with gr.Row(visible=True) as s2st_example_row:
                s2st_examples = gr.Examples(
                    examples=[
                        ["assets/sample_input.mp3", "French"],
                        ["assets/sample_input.mp3", "Mandarin Chinese"],
                        ["assets/sample_input_2.mp3", "Hindi"],
                        ["assets/sample_input_2.mp3", "Spanish"],
                    ],
                    inputs=[input_audio_file, target_language],
                    outputs=[output_audio, output_text],
                    fn=process_s2st_example,
                    cache_examples=CACHE_EXAMPLES,
                )
            with gr.Row(visible=False) as s2tt_example_row:
                s2tt_examples = gr.Examples(
                    examples=[
                        ["assets/sample_input.mp3", "French"],
                        ["assets/sample_input.mp3", "Mandarin Chinese"],
                        ["assets/sample_input_2.mp3", "Hindi"],
                        ["assets/sample_input_2.mp3", "Spanish"],
                    ],
                    inputs=[input_audio_file, target_language],
                    outputs=[output_audio, output_text],
                    fn=process_s2tt_example,
                    cache_examples=CACHE_EXAMPLES,
                )
            with gr.Row(visible=False) as t2st_example_row:
                t2st_examples = gr.Examples(
                    examples=[
                        ["My favorite animal is the elephant.", "English", "French"],
                        [
                            "My favorite animal is the elephant.",
                            "English",
                            "Mandarin Chinese",
                        ],
                        [
                            "Meta AI's Seamless M4T model is democratising spoken communication across language barriers",
                            "English",
                            "Hindi",
                        ],
                        [
                            "Meta AI's Seamless M4T model is democratising spoken communication across language barriers",
                            "English",
                            "Spanish",
                        ],
                    ],
                    inputs=[input_text, source_language, target_language],
                    outputs=[output_audio, output_text],
                    fn=process_t2st_example,
                    cache_examples=CACHE_EXAMPLES,
                )
            with gr.Row(visible=False) as t2tt_example_row:
                t2tt_examples = gr.Examples(
                    examples=[
                        ["My favorite animal is the elephant.", "English", "French"],
                        [
                            "My favorite animal is the elephant.",
                            "English",
                            "Mandarin Chinese",
                        ],
                        [
                            "Meta AI's Seamless M4T model is democratising spoken communication across language barriers",
                            "English",
                            "Hindi",
                        ],
                        [
                            "Meta AI's Seamless M4T model is democratising spoken communication across language barriers",
                            "English",
                            "Spanish",
                        ],
                    ],
                    inputs=[input_text, source_language, target_language],
                    outputs=[output_audio, output_text],
                    fn=process_t2tt_example,
                    cache_examples=CACHE_EXAMPLES,
                )
            with gr.Row(visible=False) as asr_example_row:
                asr_examples = gr.Examples(
                    examples=[
                        ["assets/sample_input.mp3", "English"],
                        ["assets/sample_input_2.mp3", "English"],
                    ],
                    inputs=[input_audio_file, target_language],
                    outputs=[output_audio, output_text],
                    fn=process_asr_example,
                    cache_examples=CACHE_EXAMPLES,
                )

            audio_source.change(
                fn=update_audio_ui,
                inputs=audio_source,
                outputs=[
                    input_audio_mic,
                    input_audio_file,
                ],
                queue=False,
                api_name=False,
            )
            task_name.change(
                fn=update_input_ui,
                inputs=task_name,
                outputs=[
                    audio_box,
                    input_text,
                    source_language,
                    target_language,
                ],
                queue=False,
                api_name=False,
            ).then(
                fn=update_output_ui,
                inputs=task_name,
                outputs=[output_audio, output_text],
                queue=False,
                api_name=False,
            ).then(
                fn=update_example_ui,
                inputs=task_name,
                outputs=[
                    s2st_example_row,
                    s2tt_example_row,
                    t2st_example_row,
                    t2tt_example_row,
                    asr_example_row,
                ],
                queue=False,
                api_name=False,
            )

            btn_load_model.click(
                fn=_load_model,
                inputs=[],
                outputs=[model_loaded],
                api_name=False,
            )

            btn.click(
                fn=_predict,
                inputs=[
                    task_name,
                    audio_source,
                    input_audio_mic,
                    input_audio_file,
                    input_text,
                    source_language,
                    target_language,
                    model_loaded,
                ],
                outputs=[output_audio, output_text],
                api_name="run",
            )

        gradio_app, local_url, share_url = demo.launch(
            share=False,
            quiet=True,
            prevent_thread_lock=True,
            server_name="0.0.0.0",
            show_error=False,
        )
        return {"share_url": share_url, "local_url": local_url}

    @mcp.tool()
    def model_trial(self, project, **kwargs):
        return {"message": "Done", "result": "Done"}

    @mcp.tool()
    def download(self, project, **kwargs):
        from flask import request, send_from_directory

        file_path = request.args.get("path")
        print(request.args)
        return send_from_directory(os.getcwd(), file_path)
