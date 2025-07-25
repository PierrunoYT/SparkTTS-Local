<div align="center">
    <h1>
    SparkTTS-Local (Windows)
    </h1>
    <p>
    <b>🪟 Windows-optimized setup for SparkTTS with CUDA acceleration</b><br>
    <em>Local Windows installation of SparkTTS text-to-speech system</em>
    </p>
    <p>
    </p>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/Original-Repository-blue" alt="original"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/Platform-Windows-lightgrey" alt="platform"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/Python-3.10%2B-orange" alt="python"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-brightgreen" alt="pytorch"></a>
</div>

## SparkTTS-Local 🔥

> **⚠️ This is a Windows-focused fork optimized for local installation and ease of use.**  
> **🚀 CUDA/GPU acceleration is strongly recommended for acceptable performance.**

### Overview

SparkTTS-Local provides a streamlined Windows setup for the SparkTTS text-to-speech system. This repository includes comprehensive installation guides and Windows-specific optimizations for running SparkTTS with CUDA acceleration.

### Key Features

- **Windows Optimized**: Windows-specific setup instructions
- **CUDA Acceleration**: Optimized for NVIDIA GPU acceleration
- **Easy Installation**: Step-by-step Windows installation guide
- **Web Interface**: Simple web UI for voice cloning and text-to-speech
- **Voice Cloning**: Zero-shot voice cloning capabilities
- **Bilingual Support**: Supports both Chinese and English

## Install

### 🛠️ Requirements

- **Windows 10/11**
- **Python 3.10 or 3.11** (recommended for best CUDA compatibility)
- **Git**
- **NVIDIA GPU with CUDA support** (strongly recommended)
- **CUDA drivers** installed on your system

> **⚠️ Performance Note:** While CPU-only mode is possible, it will be significantly slower. GPU acceleration via CUDA is highly recommended for practical use.

### 🚀 Windows Installation

#### 1. **Install Python**

- Download Python 3.11.x from [python.org](https://www.python.org/downloads/release/python-3116/)
- During install, **check "Add Python to PATH"**

#### 2. **Clone the Repository**

```sh
git clone https://github.com/PierrunoYT/SparkTTS-Local.git
cd SparkTTS-Local
```

#### 3. **Create and Activate Virtual Environment**

```sh
python -m venv venv
.\venv\Scripts\activate
```

> **💡 PowerShell Note:** If you're using PowerShell, you may need to enable script execution: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

#### 4. **Upgrade pip**

```sh
pip install --upgrade pip
```

#### 5. **Install Dependencies**

```sh
pip install -r requirements.txt
```

#### 6. **Install PyTorch with CUDA Support**

**CUDA 11.8:**

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.6:**

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**CUDA 12.8:**

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> **⚠️ CPU-Only Installation:** If you don't have an NVIDIA GPU, you can install CPU-only PyTorch with `pip install torch torchvision torchaudio`, but performance will be significantly slower.

### 📦 **Model Download**

#### Method 1: Using the provided script (Recommended)

```sh
python download_model.py
```

#### Method 2: Manual download via python:
```python
from huggingface_hub import snapshot_download

snapshot_download("SparkAudio/Spark-TTS-0.5B", local_dir="pretrained_models/Spark-TTS-0.5B")
```

#### Method 3: Download via git clone:
```sh
mkdir -p pretrained_models

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

git clone https://huggingface.co/SparkAudio/Spark-TTS-0.5B pretrained_models/Spark-TTS-0.5B
```

## Usage

### 🎯 **Quick Start**

#### Windows Web Interface:
Run the web interface:
```sh
python webui.py --device 0
```

#### Command Line Usage:
For direct command line inference:

``` sh
python -m cli.inference \
    --text "text to synthesis." \
    --device 0 \
    --save_dir "path/to/save/audio" \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --prompt_text "transcript of the prompt audio" \
    --prompt_speech_path "path/to/prompt_audio"
```

### 🌐 **Web Interface**

The web interface allows you to perform Voice Cloning and Voice Creation. Voice Cloning supports uploading reference audio or directly recording audio.

**Starting the Web UI:**
- Windows: Run `python webui.py --device 0`
- **Custom server settings:** `python webui.py --server_name 0.0.0.0 --server_port 7860 --device 0`

**Access:** The web interface will be available at [http://127.0.0.1:7860](http://127.0.0.1:7860) (local access only)

**Network Access:** To allow access from other devices on your network, use `--server_name 0.0.0.0`

**Important:** For voice cloning, you must upload or record an audio file as reference. The system will return an error if no audio file is provided.

**Voice Cloning and Voice Creation interfaces available in the Web UI**

## ⚠️ **Troubleshooting**

### Common Issues

### Voice Quality Issues

**Female voices sound male / Male voices sound like whispers**
- Solution: This is often caused by high temperature settings during generation
- The default temperature has been lowered from 0.8 to 0.6 for more stable voice generation
- If issues persist, try different reference audio files for voice cloning
- For voice creation, experiment with different pitch and speed combinations

**Poor audio quality or distorted output**
- Ensure your reference audio (for voice cloning) is high quality and at least 16kHz sample rate
- Try shorter text inputs (under 200 characters) for better results
- Check that your CUDA drivers are up to date for optimal GPU performance

### Installation Issues

**Virtual Environment Issues**
- **Python executable not found in venv:** Your virtual environment may have broken paths. Recreate it:
  ```sh
  rmdir /s venv
  python -m venv venv
  .\venv\Scripts\activate
  pip install -r requirements.txt
  ```
- **PowerShell execution policy errors:** Enable script execution with `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**`ImportError: cannot import name 'InterpolationMode' from 'torchvision.transforms'`**
- Solution: Install torchvision - `pip install torchvision`
- This package is required by transformers but not automatically installed

**`TypeError: Invalid file: None` when using Voice Cloning**
- Solution: Upload or record an audio file before running voice cloning
- Voice cloning requires a reference audio file

**`ModuleNotFoundError: No module named 'cli.SparkTTS'`**
- Solution: Use `python webui.py` for web interface or `python -m cli.inference` for command line

**`CUDA available: False` (when you have NVIDIA GPU)**
- Solution: Install CUDA-enabled PyTorch using one of the supported CUDA versions (11.8, 12.6, or 12.8) as shown in the installation section
- Check your GPU drivers are up to date

**Missing tensor warnings (mel_transformer.spectrogram.window)**
- These are normal warnings and don't affect functionality
- The tensors are loaded dynamically when needed

### Check Your Installation

Test if PyTorch can see your GPU:

```python
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name())
```

## ⚠️ Usage Disclaimer

This project provides a zero-shot voice cloning TTS model intended for academic research, educational purposes, and legitimate applications, such as personalized speech synthesis, assistive technologies, and linguistic research.

Please note:

- Do not use this model for unauthorized voice cloning, impersonation, fraud, scams, deepfakes, or any illegal activities.

- Ensure compliance with local laws and regulations when using this model and uphold ethical standards.

- The developers assume no liability for any misuse of this model.

We advocate for the responsible development and use of AI and encourage the community to uphold safety and ethical principles in AI research and applications. If you have any concerns regarding ethics or misuse, please contact us.