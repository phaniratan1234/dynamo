from datasets import load_dataset
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

# data/scripts/data
DATA_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# === config===
DATASETS = {
    "glue_sst2": {"train": 7000, "validation": 1500, "test": 1500},
    "xsum": {"train": 10500, "validation": 2250, "test": 2250},
    "squad": {"train": 14000, "validation": 3000, "test": 3000},
}
 

def sample_and_save(df: pd.DataFrame, size: int, filename: str):
    print(f"📦 Sampling {size} rows → {filename}")

    try:
        if len(df) < size:
            print(f"⚠️  Only {len(df)} available. Saving all.")
            sampled = df
        else:
            sampled = df.sample(n=size, random_state=42)

        output_path = RAW_DIR / filename
        sampled.to_json(output_path, orient="records", lines=True, force_ascii=False)
        print(f"✅ Saved: {output_path}")
    except Exception as e:
        print(f"Error while Sampling {size} rows → {filename}")


def main():
    for dataset_name, splits in DATASETS.items():
        # dataset_name = "glue/sst2"
        # splits = {"train": 7000, "validation": 1500, "test": 1500}

        parts = dataset_name.split('_')
        mainset = parts[0]
        subset = parts[1] if len(parts) > 1 else None

        print(f"\nLoading dataset: {mainset} {f'({subset})' if subset else ''}")
        try:
            if subset:
                dataset_dict = load_dataset(mainset, subset) 
            else:
                dataset_dict = load_dataset(mainset) 
        except Exception as e:
            print(f"Error while Loading dataset: {mainset} {f'({subset})' if subset else ''}")

        # squad doesnt have test set
        if dataset_name == "squad" and "test" not in dataset_dict:
            # split validation into half val/test 
            print("Splitting squad validation set into validation/test")
            val_full = pd.DataFrame(dataset_dict["validation"])
            val_half_1, val_half_2 = train_test_split(val_full, test_size=0.5, random_state=42)

            sample_and_save(pd.DataFrame(dataset_dict["train"]), splits["train"], f"train_{dataset_name}_{TIMESTAMP}.json")
            sample_and_save(val_half_1, splits["validation"], f"validation_{dataset_name}_{TIMESTAMP}.json")
            sample_and_save(val_half_2, splits["test"], f"test_{dataset_name}_{TIMESTAMP}.json")
        else:
            for set_type, size in splits.items():
                try:
                    df = pd.DataFrame(dataset_dict[set_type])
                except KeyError:
                    print(f"{dataset_name} has no split named '{set_type}'")
                    continue

                filename = f"{set_type}_{dataset_name}_{TIMESTAMP}.json"
                sample_and_save(df, size, filename)


if __name__ == '__main__':
    # train lora: SST-2 / SQuAD / XSum
    # train router: multi-task example
    # overall train: 
    main()