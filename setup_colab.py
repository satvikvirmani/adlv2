#!/usr/bin/env python3
"""
Setup script for Google Colab environment
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    requirements = [
        "torch>=1.13",
        "torchvision", 
        "tensorboard",
        "numpy==1.26.4",
        "Pillow",
        "opencv-python",
        "matplotlib",
        "tqdm"
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "--quiet", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
            return False
    
    return True

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ No GPU available, will use CPU")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def setup_colab():
    """Main setup function"""
    print("Setting up Google Colab environment for ADL training...")
    print("="*60)
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements")
        return False
    
    print("\n" + "="*60)
    
    # Check GPU
    gpu_available = check_gpu()
    
    print("\n" + "="*60)
    print("Setup completed!")
    
    if gpu_available:
        print("✓ Ready to train with GPU acceleration")
    else:
        print("⚠ Will train on CPU (slower but functional)")
    
    # print("\nNext steps:")
    # print("1. Upload your dataset to Google Drive")
    # print("2. Run: python run_train_colab.py")
    
    return True

if __name__ == "__main__":
    setup_colab() 