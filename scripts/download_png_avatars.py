#!/usr/bin/env python3
"""
Download all PNG files from ModelScope LiteAvatarGallery repository

This script downloads all PNG files from the ModelScope repository
https://www.modelscope.cn/models/HumanAIGC-Engineering/LiteAvatarGallery/files
and its subfolders, preserving the directory structure.

Usage:
    python scripts/download_png_avatars.py
"""

import os
import sys
import shutil
import subprocess as sp
import tempfile
from typing import List, Tuple

from loguru import logger


class PNGAvatarDownloader:
    """Download PNG files from ModelScope repository"""

    def __init__(self, project_root: str = None, output_dir: str = "Lite_Avatar_avatars"):
        """
        Initialize the downloader

        Args:
            project_root: Project root directory, auto-detected if None
            output_dir: Output directory name for downloaded PNG files
        """
        if project_root is None:
            # Get the parent directory of script location as project root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)

        self.project_root = project_root
        self.output_dir = os.path.join(project_root, output_dir)
        self.model_id = "HumanAIGC-Engineering/LiteAvatarGallery"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def download_all_png_files(self) -> Tuple[int, int]:
        """
        Download all PNG files from the repository

        Returns:
            Tuple of (successful_downloads, total_files)
        """
        # Use git clone method directly as it's more reliable
        return self._download_via_git()

    def _download_via_git(self) -> Tuple[int, int]:
        """
        Clone repository and copy PNG files while preserving folder structure
        
        Returns:
            Tuple of (successful_downloads, total_files)
        """
        temp_dir = tempfile.mkdtemp()
        repo_path = os.path.join(temp_dir, "LiteAvatarGallery")
        
        try:
            logger.info("Cloning repository to extract PNG files...")
            logger.info("This may take a while as the repository may be large...")
            
            # Check if git-lfs is needed
            try:
                sp.run(["git", "lfs", "version"], check=True, capture_output=True, timeout=5)
                logger.info("Git LFS found, installing...")
                sp.run(["git", "lfs", "install"], check=True, timeout=5)
            except (sp.TimeoutExpired, sp.CalledProcessError, FileNotFoundError):
                logger.warning("Git LFS not found or not working, but continuing anyway...")
            
            # Clone repository with depth 1 to speed up
            import time
            cmd = ["git", "clone", "--depth", "1", "--progress",
                   f"https://www.modelscope.cn/models/{self.model_id}.git",
                   repo_path]
            
            logger.info(f"Running: {' '.join(cmd)}")
            logger.info("Please wait, this may take several minutes...")
            logger.info("The repository may be large, so cloning can take time...")
            
            # Run git clone with timeout - use communicate() to avoid blocking
            logger.info("Starting git clone (this may take a while)...")
            process = sp.Popen(
                cmd,
                stdout=sp.PIPE,
                stderr=sp.STDOUT,
                text=True
            )
            
            # Use a thread to monitor progress
            import threading
            output_lines = []
            output_lock = threading.Lock()
            
            def read_output():
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        with output_lock:
                            output_lines.append(line)
            
            reader_thread = threading.Thread(target=read_output, daemon=True)
            reader_thread.start()
            
            # Wait for process with timeout
            start_time = time.time()
            timeout = 3600  # 60 minutes timeout
            last_log_time = start_time
            
            while process.poll() is None:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    process.kill()
                    raise TimeoutError(f"Git clone timed out after {timeout} seconds ({timeout/60:.1f} minutes)")
                
                # Log any new output
                with output_lock:
                    while output_lines:
                        line = output_lines.pop(0)
                        logger.info(f"Git: {line}")
                        last_log_time = time.time()
                
                # Log progress every 30 seconds
                if time.time() - last_log_time > 30:
                    logger.info(f"Git clone still running... ({elapsed/60:.1f} minutes elapsed)")
                    last_log_time = time.time()
                
                time.sleep(1)
            
            # Read any remaining output
            reader_thread.join(timeout=5)
            with output_lock:
                for line in output_lines:
                    logger.info(f"Git: {line}")
            
            if process.returncode != 0:
                logger.error(f"Git clone failed with return code {process.returncode}")
                raise sp.CalledProcessError(process.returncode, cmd)
            
            logger.info("Repository cloned successfully!")
            
            # Find and copy all PNG files
            logger.info("Scanning for PNG files...")
            successful = 0
            total = 0
            
            # First, count all PNG files
            png_files = []
            for root, dirs, files in os.walk(repo_path):
                for file in files:
                    if file.lower().endswith('.png'):
                        src_path = os.path.join(root, file)
                        rel_path = os.path.relpath(src_path, repo_path)
                        png_files.append((src_path, rel_path))
            
            total = len(png_files)
            logger.info(f"Found {total} PNG files. Starting copy...")
            
            # Copy files with progress
            for idx, (src_path, rel_path) in enumerate(png_files, 1):
                dst_path = os.path.join(self.output_dir, rel_path)
                dst_dir = os.path.dirname(dst_path)
                
                os.makedirs(dst_dir, exist_ok=True)
                
                try:
                    shutil.copy2(src_path, dst_path)
                    if idx % 10 == 0 or idx == total:
                        logger.info(f"Progress: {idx}/{total} - Copied: {rel_path}")
                    successful += 1
                except Exception as e:
                    logger.error(f"Failed to copy {rel_path}: {e}")
            
            logger.info(f"Downloaded {successful}/{total} PNG files")
            return successful, total
            
        except sp.CalledProcessError as e:
            logger.error(f"Git clone failed: {e.stderr}")
            return 0, 0
        except Exception as e:
            logger.error(f"Error during git-based download: {e}")
            return 0, 0
        finally:
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


def main():
    """Main function"""
    # Configure logging - also print to stdout for visibility
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    # Also add stdout handler
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO"
    )
    
    try:
        print("Starting PNG avatar download script...", file=sys.stderr)
        downloader = PNGAvatarDownloader()
        print(f"Output directory: {downloader.output_dir}", file=sys.stderr)
        logger.info(f"Starting download to: {downloader.output_dir}")
        
        successful, total = downloader.download_all_png_files()
        
        if successful > 0:
            logger.info(f"Successfully downloaded {successful}/{total} PNG files to {downloader.output_dir}")
            print(f"Successfully downloaded {successful}/{total} PNG files to {downloader.output_dir}", file=sys.stderr)
        else:
            logger.error("No files were downloaded. Please check your internet connection and ModelScope access.")
            print("No files were downloaded. Please check your internet connection and ModelScope access.", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        print(f"Operation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

