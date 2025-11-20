# LAM Avatar Guide

## Getting More Avatars

### Option 1: Download from HuggingFace

The LAM project provides pre-made avatars on HuggingFace:

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Run the download script
python scripts/download_lam_avatars.py
```

Or manually download from: https://huggingface.co/3DAIGC/LAM-assets

### Option 2: Use ModelScope Avatar Export Feature

Visit the ModelScope Space: https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model

This provides a web interface to generate and export avatars.

## Creating Your Own LAM Avatars

### Prerequisites

1. **Clone the LAM repository:**
   ```bash
   git clone https://github.com/aigc3d/LAM.git
   cd LAM
   ```

2. **Set up the environment** (follow LAM's installation instructions)

3. **Install Blender** (required for exporting avatars for interactive use)

### Steps to Create a Custom Avatar

1. **Generate Avatar from Image:**
   ```bash
   python app_lam.py
   ```
   This launches an interface where you can input an image to generate a 3D avatar.

2. **Export Avatar for OpenAvatarChat:**
   ```bash
   python app_lam.py --blender_path /path/to/blender
   ```
   Replace `/path/to/blender` with your actual Blender installation path.

3. **Place the exported .zip file:**
   - Copy your exported avatar `.zip` file to:
     ```
     src/handlers/client/h5_rendering_client/lam_samples/your_avatar.zip
     ```

4. **Update your config:**
   ```yaml
   LamClient:
     module: client/h5_rendering_client/client_handler_lam
     asset_path: "lam_samples/your_avatar.zip"
   ```

### Resources

- **LAM Project:** https://github.com/aigc3d/LAM
- **LAM-A2E (Audio to Expression):** https://github.com/aigc3d/LAM_Audio2Expression
- **ModelScope Space:** https://www.modelscope.cn/studios/Damo_XR_Lab/LAM-A2E
- **Avatar Export Feature:** https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model

### Note

According to the documentation, the ready-to-use generation pipeline for LAM avatars is still in development. The ModelScope Avatar Export Feature mentioned above may be the easiest way to create custom avatars currently.

