import pandas as pd


class Loader:
    def __init__(self):
        self.ts_df = self.ts_dataset_loader()

    def ts_dataset_loader(self, path="data/twitter/ts_dataset/training.jsonl"):
        return pd.read_json(path, lines=True)
