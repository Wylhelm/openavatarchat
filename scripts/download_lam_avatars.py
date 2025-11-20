#!/usr/bin/env python3
"""
Script to download LAM avatars from HuggingFace
"""
import os
from huggingface_hub import hf_hub_download
import tarfile

def download_lam_avatars(output_dir="src/handlers/client/h5_rendering_client/lam_samples"):
    """Download LAM avatars from HuggingFace"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading LAM assets from HuggingFace...")
    
    # Download LAM assets
    try:
        asset_file = hf_hub_download(
            repo_id="3DAIGC/LAM-assets",
            repo_type='model',
            filename='LAM_assets.tar',
            local_dir='./temp_lam_download'
        )
        
        print(f"Extracting {asset_file}...")
        with tarfile.open(asset_file, 'r') as tar:
            tar.extractall(path=output_dir)
        
        print(f"LAM avatars extracted to {output_dir}")
        print("You can now use these avatars by setting asset_path in your config.")
        
    except Exception as e:
        print(f"Error downloading: {e}")
        print("\nAlternative: You can manually download from:")
        print("https://huggingface.co/3DAIGC/LAM-assets")
        print("\nOr use the ModelScope Avatar Export Feature:")
        print("https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model")

if __name__ == "__main__":
    download_lam_avatars()

