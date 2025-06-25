# 🚀 DYNAMO Cloud Deployment Guide

## Option 1: Google Colab (FREE - Recommended)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create new notebook
3. Runtime → Change runtime type → GPU (T4)
4. Copy and run:

```python
# Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers datasets accelerate wandb

# Clone project
!git clone https://github.com/YOUR_USERNAME/dynamo_project.git
%cd dynamo_project

# Quick test
!python train.py --config config_working.yaml --phase 1 --device cuda

# Full training
!python train.py --config config_cloud.yaml --phase 1 --device cuda
```

## Option 2: Kaggle Notebooks (FREE)

1. Go to [Kaggle Code](https://www.kaggle.com/code)
2. New Notebook → GPU T4 x2
3. Copy colab code above (replace %cd with os.chdir)

## Option 3: Vast.ai ($0.20/hour)

1. Sign up at [Vast.ai](https://vast.ai/)
2. Search for "pytorch" image with RTX 3090+
3. SSH and run:

```bash
git clone https://github.com/YOUR_USERNAME/dynamo_project.git
cd dynamo_project
pip install -r requirements.txt
python train.py --config config_cloud.yaml --phase 1 --device cuda
```

## Upload Steps

1. Create GitHub repo:
```bash
git init
git add .
git commit -m "Initial DYNAMO implementation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/dynamo_project.git
git push -u origin main
```

2. Replace YOUR_USERNAME in the URLs above

## Training Configs

- `config_working.yaml`: Small datasets for testing
- `config_cloud.yaml`: Optimized for cloud GPU training  
- `config.yaml`: Original full-scale training

## Expected Results

- **Phase 1**: ~1-2 hours on T4 GPU
- **Model size**: ~25M parameters (much smaller now)
- **Memory**: ~4GB GPU memory
