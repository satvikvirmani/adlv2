import subprocess

EXPERIMENT = "rgbham10000v1"
CHANNELS_NUM = 3

cmd = [
    "python3", "train.py",
    "--DENOISER", "efficient_Unet",
    "--num-workers", "2",
    "--EXPERIMENT", EXPERIMENT,
    "--json-file", "../configs/ADL_train.json",
    "--CHANNELS-NUM", str(CHANNELS_NUM),
    "--train-dirs", "/content/drive/MyDrive/adlv2_ham10000/train",
    "--test-dirs", "/content/drive/MyDrive/adlv2_ham10000/test"
]
subprocess.run(cmd)