#!/usr/bin/env python3
"""
Create GitHub Repository for DYNAMO Project
Upload to GitHub for cloud GPU training.
"""

import os
import subprocess

def create_gitignore():
    """Create comprehensive .gitignore file."""
    gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# PyTorch
*.pth
*.pt

# Datasets and cache
cache/
outputs/
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
fix_*.py
setup_*.py
*_setup.txt
'''
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("✅ Created .gitignore")

def create_github_workflow():
    """Create GitHub Actions workflow for cloud training."""
    workflow_dir = '.github/workflows'
    os.makedirs(workflow_dir, exist_ok=True)
    
    workflow_content = '''name: Train DYNAMO on GPU

on:
  workflow_dispatch:  # Manual trigger
  push:
    branches: [ main ]

jobs:
  train:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Train Phase 1
      run: |
        python train.py --config config_working.yaml --phase 1 --device cpu
    
    - name: Upload checkpoints
      uses: actions/upload-artifact@v3
      with:
        name: dynamo-checkpoints
        path: checkpoints/
'''
    
    with open(os.path.join(workflow_dir, 'train.yml'), 'w') as f:
        f.write(workflow_content)
    print("✅ Created GitHub Actions workflow")

def create_deployment_guide():
    """Create deployment guide for different platforms."""
    guide = '''# 🚀 DYNAMO Cloud Deployment Guide

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
'''
    
    with open('DEPLOYMENT.md', 'w') as f:
        f.write(guide)
    print("✅ Created DEPLOYMENT.md guide")

def initialize_git():
    """Initialize git repository."""
    commands = [
        "git init",
        "git add .",
        "git commit -m 'Initial DYNAMO implementation with cloud training support'"
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {cmd}")
            else:
                print(f"⚠️  {cmd}: {result.stderr}")
        except Exception as e:
            print(f"❌ {cmd}: {e}")

if __name__ == "__main__":
    print("📦 PREPARING DYNAMO FOR GITHUB")
    print("=" * 40)
    
    create_gitignore()
    create_github_workflow()
    create_deployment_guide()
    
    print("\n🔧 Initializing Git...")
    initialize_git()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Create GitHub repository")
    print("2. git remote add origin https://github.com/YOUR_USERNAME/dynamo_project.git")
    print("3. git push -u origin main")
    print("4. Use DEPLOYMENT.md guide for cloud training")
    print("\n✅ Ready for cloud deployment!") 