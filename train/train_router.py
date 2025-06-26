from pathlib import Path
from datetime import datetime

# data/scripts/data
DATA_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = DATA_DIR /'processed'
PROCESSED_DIR .mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")




def train_router():
    dataset =  RouterDataset(config["data_path"])
    print()

if __name__ == '__main__':
    config = {
        "data_path": f'{PROCESSED_DIR}',
        "epochs": 3,
        "batch_size": 16,
        "lr": 1e-4,
        "input_dim": 768,
        "hidden_dim": 256,
        "save_path": "router.pth"
    }
    train_router(config)