import pandas as pd
from sklearn.model_selection import train_test_split


def load_json_dataset(path="data/headlines_dataset/Sarcasm_Headlines_Dataset.json"):
    data = pd.read_json(path, lines=True)
    data = data.drop(["article_link"], axis=1)

    train_val, test_data = train_test_split(data, test_size=0.1, random_state=42)
    train_data, val_data = train_test_split(train_val, test_size=0.11, random_state=42)

    return train_data, val_data, test_data
