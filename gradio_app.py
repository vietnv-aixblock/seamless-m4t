from __future__ import annotations

import os

import gradio as gr
import numpy as np
import torchaudio
from transformers import AutoProcessor, SeamlessM4TModel

from lang_list import (
    LANG_TO_SPKR_ID,
    LANGUAGE_NAME_TO_CODE,
    S2ST_TARGET_LANGUAGE_NAMES,
    S2TT_TARGET_LANGUAGE_NAMES,
    T2TT_TARGET_LANGUAGE_NAMES,
    TEXT_SOURCE_LANGUAGE_NAMES,
)

# ------------------------------------------------------------------------------------------------------------------------------
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


def predict(
    task_name: str,
    audio_source: str,
    input_audio_mic: str | None,
    input_audio_file: str | None,
    input_text: str | None,
    source_language: str | None,
    target_language: str,
    model_demo: SeamlessM4TModel,
    processor_demo: AutoProcessor,
    device: str,
) -> tuple[tuple[int, np.ndarray] | None, str]:
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


def process_s2st_example(
    input_audio_file: str, target_language: str
) -> tuple[tuple[int, np.ndarray] | None, str]:
    return predict(
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
    return predict(
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
    return predict(
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
    return predict(
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
    return predict(
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
