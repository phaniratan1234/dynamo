# 🚀 DYNAMO Kaggle Deployment Guide

## Step 1: Create GitHub Repository (2 minutes)

1. Go to: https://github.com/new
2. Repository name: `dynamo_project`
3. Set to Public (required for Kaggle access)
4. Click "Create repository"
5. Copy the repository URL (you'll need this)

## Step 2: Upload Your Code (1 minute)

Run these commands in your terminal:

```bash
# Add your GitHub repository URL here
git remote add origin https://github.com/YOUR_USERNAME/dynamo_project.git
git push -u origin main
```

## Step 3: Setup Kaggle Notebook (3 minutes)

1. Go to: https://www.kaggle.com/code
2. Click "New Notebook"
3. Settings → Accelerator → **GPU T4 x2** (IMPORTANT!)
4. Settings → Internet → **On** (to download datasets)
5. Copy and paste this code:

```python
# Kaggle DYNAMO Training Setup
import os
import subprocess

# Clone your project
!git clone https://github.com/YOUR_USERNAME/dynamo_project.git
os.chdir('/kaggle/working/dynamo_project')

# Install additional packages
!pip install accelerate wandb

# Verify GPU setup
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Quick validation test
print("🔍 Running quick validation...")
!python validate_project.py

# Start Phase 1 training with working config (small datasets)
print("🚀 Starting Phase 1 training...")
!python train.py --config config_working.yaml --phase 1 --device cuda

# If that works, run full training
print("🎯 Starting full Phase 1 training...")
!python train.py --config config_cloud.yaml --phase 1 --device cuda
```

## Step 4: Monitor Training

- Training should start automatically
- Phase 1 with working config: ~10 minutes
- Phase 1 with cloud config: ~1-2 hours
- Monitor GPU usage in Kaggle's system panel

## Step 5: Download Results

Add this at the end of your notebook:

```python
# Download checkpoints
import shutil
shutil.make_archive('dynamo_checkpoints', 'zip', 'checkpoints')
print("✅ Checkpoints saved as dynamo_checkpoints.zip")

# Show final results
!ls -la checkpoints/phase1/
```

## Expected Output

```
✅ CUDA available: True
✅ GPU count: 2
✅ GPU 0: Tesla T4
✅ GPU 1: Tesla T4
🔍 Running quick validation...
✅ All components validated successfully
🚀 Starting Phase 1 training...
✅ Phase 1 training completed
```

## Troubleshooting

- **GPU not available**: Check accelerator settings
- **Clone fails**: Make sure repository is public
- **Import errors**: Check all files uploaded correctly
- **CUDA errors**: Restart notebook and try again

Replace `YOUR_USERNAME` with your actual GitHub username! 