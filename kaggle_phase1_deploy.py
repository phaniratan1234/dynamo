#!/usr/bin/env python3
"""
🚀 DYNAMO Phase 1 Training - Kaggle Deployment
Copy-paste this entire cell into Kaggle Notebook and run!

Requirements:
- Kaggle Notebook with GPU T4 x2
- Internet enabled
"""

# ===== KAGGLE SETUP =====
import os
import subprocess
import sys

print("🚀 DYNAMO Phase 1 Training - Starting deployment...")

# Clone repository
print("\n📥 Cloning DYNAMO repository...")
if os.path.exists("dynamo"):
    print("Repository already exists, pulling latest changes...")
    os.chdir("dynamo")
    subprocess.run(["git", "pull"], check=True)
else:
    subprocess.run(["git", "clone", "https://github.com/phaniratan1234/dynamo.git"], check=True)
    os.chdir("dynamo")

# Install dependencies
print("\n📦 Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
               "transformers", "torch", "datasets", "wandb", "rouge-score", "sacrebleu"], 
               check=True)

# Check GPU availability
print("\n🔍 Checking GPU availability...")
import torch
print(f"✅ CUDA available: {torch.cuda.is_available()}")
print(f"✅ GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

# Verify configuration
print("\n⚙️ Verifying configuration...")
from utils.config import get_config
config = get_config("config_cloud.yaml")
print(f"✅ Config loaded: {len(config.model.lora_configs)} tasks")
print(f"✅ Training epochs: {config.training.phase1.num_epochs}")
print(f"✅ Dataset sizes: SST-2: {config.data.max_train_samples}")

# START PHASE 1 TRAINING
print("\n" + "="*60)
print("🏁 STARTING PHASE 1 TRAINING - NO TEST RUNS!")
print("Expected duration: ~25 minutes on T4 x2 GPUs (2 epochs each)")
print("="*60)

# Run Phase 1 training
try:
    subprocess.run([sys.executable, "train.py", 
                   "--config", "config_cloud.yaml", 
                   "--phase", "1"], 
                   check=True)
    
    print("\n" + "="*60)
    print("🎉 PHASE 1 TRAINING COMPLETED SUCCESSFULLY!")
    print("✅ 5 LoRA adapters trained and saved")
    print("✅ Checkpoints available in ./checkpoints/phase1/")
    print("="*60)
    
    # Create download zip
    print("\n📦 Creating download package...")
    import shutil
    shutil.make_archive('/kaggle/working/dynamo_phase1_results', 'zip', './checkpoints/phase1/')
    print("✅ Download: /kaggle/working/dynamo_phase1_results.zip")
    
    # Display file link for download
    from IPython.display import FileLink, display
    display(FileLink('/kaggle/working/dynamo_phase1_results.zip'))
    
except subprocess.CalledProcessError as e:
    print(f"\n❌ Training failed with error: {e}")
    print("Check logs above for details")
    
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    print("Please check the error logs above")

print("\n🏁 Deployment script completed!") 