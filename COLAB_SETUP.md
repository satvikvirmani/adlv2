# Google Colab Setup Guide for ADL Training

This guide will help you set up and run the ADL (Adversarial Denoising Learning) training on Google Colab.

## Prerequisites

1. **Google Colab Account**: Make sure you have access to Google Colab
2. **Google Drive**: Your dataset should be uploaded to Google Drive
3. **Dataset Structure**: Ensure your dataset follows the expected structure

## Step-by-Step Setup

### 1. Upload Your Code to Colab

First, upload your codebase to Google Colab. You can either:
- Upload the entire `codebase` folder to Colab
- Or clone from a repository if you have one

### 2. Upload Your Dataset

Upload your HAM10000 dataset to Google Drive with this structure:
```
/content/drive/MyDrive/adlv2_ham10000/
├── train/
│   ├── ISIC_0024347.jpg
│   ├── ISIC_0024350.jpg
│   └── ... (all training images)
└── test/
    ├── ISIC_0024340.jpg
    ├── ISIC_0024341.jpg
    └── ... (all test images)
```

### 3. Install Dependencies

Run the setup script to install all required packages:

```python
# In a Colab cell
!python setup_colab.py
```

### 4. Verify GPU Availability

Check if GPU is available (recommended for faster training):

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 5. Run Training

Execute the training script:

```python
!python run_train_colab.py
```

## Configuration

The training uses the configuration in `configs/ADL_train.json`. Key settings:

- **Batch Size**: 2 (reduced for Colab memory constraints)
- **Image Size**: 512x512
- **Epochs**: 50 for each phase
- **Learning Rate**: 1e-4 for denoiser, 5e-5 for ADL

## Training Phases

The training consists of three phases:

1. **Phase 1**: Warm-up denoiser training
2. **Phase 2**: Discriminator training  
3. **Phase 3**: Full ADL training

## Monitoring Training

Training progress will be displayed in the console. You can also use TensorBoard:

```python
# In a separate cell to monitor training
%load_ext tensorboard
%tensorboard --logdir ./logs
```

## Troubleshooting

### Common Issues:

1. **"Device index must not be negative"**: ✅ **FIXED** - This was resolved by updating the loss function to handle CPU tensors properly.

2. **Out of Memory**: 
   - Reduce batch size in `configs/ADL_train.json`
   - Use smaller image size
   - Restart runtime and try again

3. **Dataset not found**:
   - Verify dataset path in Google Drive
   - Check file permissions
   - Ensure correct directory structure

4. **Slow training**:
   - Enable GPU runtime in Colab
   - Reduce batch size if needed
   - Use smaller image size for faster iteration

### Memory Optimization

For Colab's limited memory, consider these adjustments in `configs/ADL_train.json`:

```json
{
    "data": {
        "batch_size": 1,  // Reduce from 2
        "H": 256,         // Reduce from 512
        "W": 256          // Reduce from 512
    }
}
```

## Expected Output

Successful training will show:
```
configuration ********************
model ADL
data {'H': 512, 'W': 512, 'batch_size': 2, ...}
...
[i] Using 1 GPUs for training  // or 0 for CPU
Let`s start training the model ...
Train size: 659 batches
Val size: 35 batches 
Test size: 100 batches
[i] denoiser-GPU0: Creating a new model.
[i] Configuring denoiser...
[i] Compiling denoiser...
[i] Fitting denoiser...
```

## Tips for Colab

1. **Runtime Type**: Use GPU runtime for faster training
2. **Disconnect Handling**: Colab may disconnect after 12 hours - save checkpoints regularly
3. **Memory Management**: Monitor memory usage and restart if needed
4. **Save Results**: Download important files before disconnecting

## File Structure After Training

After successful training, you'll have:
```
./logs/
├── denoiser/
├── discriminator/
└── adl/
```

These contain TensorBoard logs and model checkpoints. 