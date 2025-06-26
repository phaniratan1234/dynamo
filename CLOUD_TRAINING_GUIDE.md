# 🚀 DYNAMO Cloud Training - IMMEDIATE DEPLOYMENT

## Quick Start (No Test Runs)

### 1. Kaggle Notebooks (RECOMMENDED - FREE T4 x2 GPUs)

**Step 1**: Go to [Kaggle Notebooks](https://www.kaggle.com/code)
**Step 2**: Create New Notebook → Select **GPU T4 x2** → **Internet ON**
**Step 3**: Upload your GitHub repository:

```bash
# Clone repository
!git clone https://github.com/phaniratan1234/dynamo.git
%cd dynamo

# Install dependencies  
!pip install -q transformers torch datasets wandb rouge-score sacrebleu

# IMMEDIATE Phase 1 Training (10 epochs)
!python train.py --config config_cloud.yaml --phase 1
```

### 2. Google Colab (Alternative)

**Step 1**: Open [Google Colab](https://colab.research.google.com/)
**Step 2**: Runtime → Change Runtime Type → **GPU T4** → Save
**Step 3**: Run:

```bash
# Setup
!git clone https://github.com/phaniratan1234/dynamo.git
%cd dynamo
!pip install -q transformers torch datasets wandb rouge-score sacrebleu

# DIRECT Phase 1 Training
!python train.py --config config_cloud.yaml --phase 1
```

### 3. Vast.ai (Paid - Fastest)

**Step 1**: Create account at [vast.ai](https://vast.ai/)
**Step 2**: Rent RTX 3090+ instance
**Step 3**: Connect via SSH and run:

```bash
git clone https://github.com/phaniratan1234/dynamo.git
cd dynamo
pip install transformers torch datasets wandb rouge-score sacrebleu
python train.py --config config_cloud.yaml --phase 1
```

## Training Configuration

### Phase 1 Only (Current Request)
- **Command**: `python train.py --config config_cloud.yaml --phase 1`
- **Duration**: ~25 minutes on T4 x2 GPUs (2 epochs each)
- **Output**: 5 trained LoRA adapters saved to `./checkpoints/phase1/`

### Full Pipeline (Optional)
- **Command**: `python train.py --config config_cloud.yaml --phase all`
- **Duration**: ~1 hour total (Phase 1: 25min, Phase 2: 15min, Phase 3: 20min)
- **Output**: Complete DYNAMO model with trained router

## Expected Output

```
2025-06-26 XX:XX:XX - Phase 1 training started
2025-06-26 XX:XX:XX - Training sentiment adapter (10 epochs)...
2025-06-26 XX:XX:XX - Training qa adapter (10 epochs)...
2025-06-26 XX:XX:XX - Training summarization adapter (10 epochs)...
2025-06-26 XX:XX:XX - Training code_generation adapter (10 epochs)...
2025-06-26 XX:XX:XX - Training translation adapter (10 epochs)...
2025-06-26 XX:XX:XX - Phase 1 completed! Checkpoints saved.
```

## Model Specifications
- **Parameters**: 5.9M trainable (optimized)
- **Datasets**: 10K SST-2, 20K SQuAD, 15K CNN/DM, 374 MBPP, 12K Translation
- **Epochs**: 2 per adapter (quick training)
- **Memory**: ~4GB GPU memory required

## Troubleshooting

### If training stops:
```bash
# Resume Phase 1 training
!python train.py --config config_cloud.yaml --phase 1 --resume ./checkpoints/phase1/
```

### If out of memory:
```bash
# Use smaller batch size
!python train.py --config config_working.yaml --phase 1
```

### Download results:
```python
# In Kaggle/Colab notebook
from IPython.display import FileLink
import shutil

# Zip results
shutil.make_archive('dynamo_phase1_results', 'zip', './checkpoints/phase1/')
FileLink('dynamo_phase1_results.zip')
```

## IMMEDIATE ACTION ITEMS

1. **Choose platform**: Kaggle (free) or Vast.ai (paid, faster)
2. **Copy-paste command**: `!python train.py --config config_cloud.yaml --phase 1`
3. **Wait ~25 minutes**: Phase 1 training will complete automatically
4. **Download checkpoints**: 5 trained LoRA adapters ready for Phase 2

🎯 **Ready for immediate deployment!** No test runs, no validation - straight to Phase 1 training. 