# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import soundfile as sf
import logging
import argparse
import gradio as gr
import platform

from datetime import datetime
from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI


def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device=0):
    """Load the model once at the beginning."""
    logging.info(f"Loading model from: {model_dir}")

    # Determine appropriate device based on platform and availability
    if platform.system() == "Darwin":
        # macOS with MPS support (Apple Silicon)
        device = torch.device(f"mps:{device}")
        logging.info(f"Using MPS device: {device}")
    elif torch.cuda.is_available():
        # System with CUDA support
        device = torch.device(f"cuda:{device}")
        logging.info(f"Using CUDA device: {device}")
    else:
        # Fall back to CPU
        device = torch.device("cpu")
        logging.info("GPU acceleration not available, using CPU")

    model = SparkTTS(model_dir, device)
    return model


def run_tts(
    text,
    model,
    prompt_text=None,
    prompt_speech=None,
    gender=None,
    pitch=None,
    speed=None,
    temperature=0.6,
    top_k=30,
    top_p=0.9,
    save_dir="example/results",
):
    """Perform TTS inference and save the generated audio."""
    logging.info(f"Saving audio to: {save_dir}")

    if prompt_text is not None:
        prompt_text = None if len(prompt_text) <= 1 else prompt_text

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")

    logging.info("Starting inference...")

    # Perform inference and save the output audio
    with torch.no_grad():
        wav = model.inference(
            text,
            prompt_speech,
            prompt_text,
            gender,
            pitch,
            speed,
            temperature,
            top_k,
            top_p,
        )

        sf.write(save_path, wav, samplerate=16000)

    logging.info(f"Audio saved at: {save_path}")

    return save_path


def build_ui(model_dir, device=0):

    # Initialize model
    model = initialize_model(model_dir, device=device)

    # Define callback function for voice cloning
    def voice_clone(text, prompt_text, prompt_wav_upload, prompt_wav_record, temperature, top_k, top_p):
        """
        Gradio callback to clone voice using text and optional prompt speech.
        - text: The input text to be synthesised.
        - prompt_text: Additional textual info for the prompt (optional).
        - prompt_wav_upload/prompt_wav_record: Audio files used as reference.
        - temperature, top_k, top_p: Generation parameters.
        """
        prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
        prompt_text_clean = None if len(prompt_text) < 2 else prompt_text

        if prompt_speech is None:
            return "Error: Please upload or record an audio file for voice cloning."

        audio_output_path = run_tts(
            text,
            model,
            prompt_text=prompt_text_clean,
            prompt_speech=prompt_speech,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        return audio_output_path

    # Define callback function for creating new voices
    def voice_creation(text, gender, pitch, speed, temperature, top_k, top_p):
        """
        Gradio callback to create a synthetic voice with adjustable parameters.
        - text: The input text for synthesis.
        - gender: 'male' or 'female'.
        - pitch/speed: Ranges mapped by LEVELS_MAP_UI.
        - temperature, top_k, top_p: Generation parameters.
        """
        pitch_val = LEVELS_MAP_UI[int(pitch)]
        speed_val = LEVELS_MAP_UI[int(speed)]
        audio_output_path = run_tts(
            text,
            model,
            gender=gender,
            pitch=pitch_val,
            speed=speed_val,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        return audio_output_path

    with gr.Blocks() as demo:
        # Use HTML for centered title
        gr.HTML('<h1 style="text-align: center;">Spark-TTS by SparkAudio</h1>')
        with gr.Tabs():
            # Voice Clone Tab
            with gr.TabItem("Voice Clone"):
                gr.Markdown(
                    "### Upload reference audio or recording （上传参考音频或者录音）"
                )

                with gr.Row():
                    prompt_wav_upload = gr.Audio(
                        sources="upload",
                        type="filepath",
                        label="Choose the prompt audio file, ensuring the sampling rate is no lower than 16kHz.",
                    )
                    prompt_wav_record = gr.Audio(
                        sources="microphone",
                        type="filepath",
                        label="Record the prompt audio file.",
                    )

                with gr.Row():
                    text_input = gr.Textbox(
                        label="Text", lines=3, placeholder="Enter text here"
                    )
                    prompt_text_input = gr.Textbox(
                        label="Text of prompt speech (Optional; recommended for cloning in the same language.)",
                        lines=3,
                        placeholder="Enter text of the prompt speech.",
                    )

                # Advanced Settings for Voice Clone
                with gr.Accordion("🔧 Advanced Settings", open=False):
                    gr.Markdown("**Generation Parameters** - Adjust these to control voice quality and stability")
                    with gr.Row():
                        temperature_clone = gr.Slider(
                            minimum=0.1, maximum=1.5, step=0.1, value=0.6,
                            label="🌡️ Temperature",
                            info="Lower = more stable, Higher = more creative (0.1-1.5)"
                        )
                        top_k_clone = gr.Slider(
                            minimum=5, maximum=100, step=5, value=30,
                            label="🔝 Top-K",
                            info="Number of top tokens to consider (5-100)"
                        )
                        top_p_clone = gr.Slider(
                            minimum=0.1, maximum=1.0, step=0.05, value=0.9,
                            label="🎯 Top-P",
                            info="Nucleus sampling threshold (0.1-1.0)"
                        )

                audio_output = gr.Audio(
                    label="Generated Audio", autoplay=True, streaming=True
                )

                generate_buttom_clone = gr.Button("Generate")

                generate_buttom_clone.click(
                    voice_clone,
                    inputs=[
                        text_input,
                        prompt_text_input,
                        prompt_wav_upload,
                        prompt_wav_record,
                        temperature_clone,
                        top_k_clone,
                        top_p_clone,
                    ],
                    outputs=[audio_output],
                )

            # Voice Creation Tab
            with gr.TabItem("Voice Creation"):
                gr.Markdown(
                    "### Create your own voice based on the following parameters"
                )

                with gr.Row():
                    with gr.Column():
                        gender = gr.Radio(
                            choices=["male", "female"], value="male", label="Gender"
                        )
                        pitch = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3, label="Pitch"
                        )
                        speed = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3, label="Speed"
                        )
                    with gr.Column():
                        text_input_creation = gr.Textbox(
                            label="Input Text",
                            lines=3,
                            placeholder="Enter text here",
                            value="You can generate a customized voice by adjusting parameters such as pitch and speed.",
                        )
                        create_button = gr.Button("Create Voice")

                # Advanced Settings for Voice Creation
                with gr.Accordion("🔧 Advanced Settings", open=False):
                    gr.Markdown("**Generation Parameters** - Adjust these to control voice quality and stability")
                    with gr.Row():
                        temperature_creation = gr.Slider(
                            minimum=0.1, maximum=1.5, step=0.1, value=0.6,
                            label="🌡️ Temperature",
                            info="Lower = more stable, Higher = more creative (0.1-1.5)"
                        )
                        top_k_creation = gr.Slider(
                            minimum=5, maximum=100, step=5, value=30,
                            label="🔝 Top-K",
                            info="Number of top tokens to consider (5-100)"
                        )
                        top_p_creation = gr.Slider(
                            minimum=0.1, maximum=1.0, step=0.05, value=0.9,
                            label="🎯 Top-P",
                            info="Nucleus sampling threshold (0.1-1.0)"
                        )

                audio_output = gr.Audio(
                    label="Generated Audio", autoplay=True, streaming=True
                )
                create_button.click(
                    voice_creation,
                    inputs=[text_input_creation, gender, pitch, speed, temperature_creation, top_k_creation, top_p_creation],
                    outputs=[audio_output],
                )

    return demo


def parse_arguments():
    """
    Parse command-line arguments such as model directory and device ID.
    """
    parser = argparse.ArgumentParser(description="Spark TTS Gradio server.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="ID of the GPU device to use (e.g., 0 for cuda:0)."
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="127.0.0.1",
        help="Server host/IP for Gradio app (default: 127.0.0.1 for local access only)."
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Server port for Gradio app."
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Build the Gradio demo by specifying the model directory and GPU device
    demo = build_ui(
        model_dir=args.model_dir,
        device=args.device
    )

    # Launch Gradio with the specified server name and port
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port
    )