# Open Avatar Chat

A modular, interactive digital human dialog implementation.

<p align="center" style="display: flex; flex-direction: row; justify-content: center">
 🤗 <a href="https://huggingface.co/spaces/HumanAIGC-Engineering-Team/open-avatar-chat">Demo</a>&nbsp&nbsp|&nbsp&nbsp<img alt="Static Badge" style="height: 10px;" src="./assets/images/modelscope_logo.png"> <a href="https://www.modelscope.cn/studios/HumanAIGC-Engineering/open-avatar-chat">Demo</a>&nbsp&nbsp|&nbsp&nbsp💬 <a href="https://github.com/HumanAIGC-Engineering/OpenAvatarChat/blob/main/assets/images/community_wechat.png">WeChat (微信)</a>
</p>

## 🔥 Key Features
- **Multi-modal Language Models**: Supports text, audio, and video inputs.
- **Modular Design**: Easily swap components (VAD, ASR, LLM, TTS, Avatar) to fit your needs.
- **Various Avatars Supported**: LAM (3D Gaussian Splatting), LiteAvatar (2D), MuseTalk (2D).
- **Multiple TTS Options**: XTTS, CosyVoice, EdgeTTS, ElevenLabs.

## 📋 Prerequisites
- **NVIDIA GPU** with CUDA 12.4+ support (CUDA 12.8 used in our unified Docker image).
- **Docker** (Recommended for easiest deployment).
- **Python 3.11** (If running locally).
- `git-lfs` for downloading model weights.

## 🚀 Complete Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/HumanAIGC-Engineering/OpenAvatarChat.git
cd OpenAvatarChat

# Initialize and download git submodules
git submodule update --init --recursive --depth 1
```

### 2. Download Model Weights
Depending on which avatar modules you plan to use, you'll need to download their respective weights. Run these from the project root:

**For LiteAvatar:**
```bash
bash scripts/download_liteavatar_weights.sh
```

**For LAM (Large Avatar Model):**
```bash
# Download wav2vec2-base-960h
git clone --depth 1 https://huggingface.co/facebook/wav2vec2-base-960h ./models/wav2vec2-base-960h

# Download LAM audio2exp model
wget https://huggingface.co/3DAIGC/LAM_audio2exp/resolve/main/LAM_audio2exp_streaming.tar -P ./models/LAM_audio2exp/
tar -xzvf ./models/LAM_audio2exp/LAM_audio2exp_streaming.tar -C ./models/LAM_audio2exp && rm ./models/LAM_audio2exp/LAM_audio2exp_streaming.tar
```

**For MuseTalk:**
```bash
bash scripts/download_musetalk_weights.sh
```

### 3. Deploy Using Docker (Highly Recommended)
We provide a unified Docker setup (`Dockerfile.cuda12.8`) that packages all dependencies for every possible configuration. This makes it incredibly easy to switch between configurations without rebuilding the image!

**Step 1: Build the comprehensive Docker image**
*(You only need to do this once)*
```bash
bash build_cuda128.sh
```

**Step 2: Set up Environment Variables (Optional)**
If your configuration uses cloud APIs (like Bailian, OpenAI, or an Ollama endpoint), create a `.env` file in the project root:
```bash
touch .env
# Example content:
# DASHSCOPE_API_KEY=sk-xxxxx
# OPENAI_API_KEY=sk-xxxxx
```

**Step 3: Run the Container**
You can run the container and specify *any* configuration file on the fly:
```bash
bash run_docker_cuda128.sh --config config/chat_with_lam_inessa_xtts.yaml
```

**Switching Configurations:**
To easily switch configs, simply stop the container (`CTRL+C`) and run the script again with a different YAML file:
```bash
bash run_docker_cuda128.sh --config config/chat_with_openai_compatible.yaml
```

### 4. Running Locally (Without Docker)
If you prefer running directly on your host machine without Docker:

```bash
# 1. Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create a virtual environment and install pip
uv venv --python 3.11.11
uv pip install setuptools pip

# 3. Install dependencies for your specific config
uv run install.py --uv --config config/chat_with_lam_inessa_xtts.yaml

# 4. Run post-install scripts (e.g., configuring CUDA paths)
./scripts/post_config_install.sh --config config/chat_with_lam_inessa_xtts.yaml

# 5. Start the application
uv run src/demo.py --config config/chat_with_lam_inessa_xtts.yaml
```

## 🌐 Network Configuration (SSL & WebRTC)
Since this project uses WebRTC for real-time video/audio streaming, **HTTPS is required** for the browser to grant access to your camera and microphone.

- **Local Access:** Accessing the UI via `https://127.0.0.1:8282` or `https://localhost:8282` will work seamlessly.
- **Remote Access (LAN/Internet):** If accessing from another device (like your phone or another PC):
  1. You must configure valid SSL certificates. You can generate self-signed certificates by running:
     ```bash
     bash scripts/create_ssl_certs.sh
     ```
  2. **TURN Server:** If you encounter a perpetual "waiting" state when connecting remotely, your network requires a TURN server for NAT traversal. Run `bash scripts/setup_coturn.sh` to set one up, and update your YAML configuration accordingly.

## ⚙️ Understanding Configurations
Configurations are located in the `config/` directory. They define which modules handle each step of the pipeline:
- **Client**: Handles rendering (e.g., Server-side RTC vs Client-side H5 for LAM).
- **VAD**: Voice Activity Detection (e.g., Silero).
- **ASR**: Automatic Speech Recognition (e.g., Whisper, SenseVoice).
- **LLM**: Language Model (e.g., OpenAI compatible, MiniCPM, local Ollama).
- **TTS**: Text-to-Speech (e.g., XTTS, CosyVoice, EdgeTTS).
- **Avatar**: Digital Human rendering (e.g., LAM, LiteAvatar, MuseTalk).

*Note on XTTS:* The default `XTTS` handler loads the model locally into GPU memory. If you want to use an external API (like a remote Coqui XTTS container), you can easily create a new handler in `src/handlers/tts/` based on the existing `elevenlabs` or `openai` examples.