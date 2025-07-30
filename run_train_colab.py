import subprocess
import os
import sys

# Check if we're in Google Colab
def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

# Setup for Google Colab
if is_colab():
    print("Running in Google Colab environment...")
    
    # Mount Google Drive if not already mounted
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted successfully!")
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        print("Please mount Google Drive manually if needed.")

EXPERIMENT = "rgbham10000v1"
CHANNELS_NUM = 3

# Adjust paths for Colab environment
train_dir = "/content/drive/MyDrive/adlv2_ham10000/train"
test_dir = "/content/drive/MyDrive/adlv2_ham10000/test"

# Check if directories exist
if not os.path.exists(train_dir):
    print(f"Warning: Training directory not found at {train_dir}")
    print("Please ensure your dataset is uploaded to Google Drive")
    
if not os.path.exists(test_dir):
    print(f"Warning: Test directory not found at {test_dir}")
    print("Please ensure your dataset is uploaded to Google Drive")

cmd = [
    "python3", "train.py",
    "--DENOISER", "efficient_Unet",
    "--num-workers", "2",
    "--EXPERIMENT", EXPERIMENT,
    "--json-file", "configs/ADL_train.json",
    "--CHANNELS-NUM", str(CHANNELS_NUM),
    "--train-dirs", train_dir,
    "--test-dirs", test_dir
]

print("Starting training with command:")
print(" ".join(cmd))
print("\n" + "="*50)
print("Training will begin...")
print("="*50)

try:
    result = subprocess.run(cmd, check=True)
    print("Training completed successfully!")
except subprocess.CalledProcessError as e:
    print(f"Training failed with error code: {e.returncode}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error during training: {e}")
    sys.exit(1) 