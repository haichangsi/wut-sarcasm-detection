from loader import *
from hashtag import *


class Tester:
    def __init__(self):
        self.loader = Loader()

    def load_headlines(self):
        self.df = self.loader.ts_dataset_loader(
            "data/headlines_dataset/Sarcasm_Headlines_Dataset.json"
        )
        # self.headlines = self.df["headline"]
        self.hashtag_headlines = self.df[self.df.headline.str.contains("#")]
        self.hashtag_headlines_labeled = list(
            self.hashtag_headlines[["headline", "is_sarcastic"]].itertuples(
                index=False, name=None
            )
        )

    def test_hashtag_on_headlines(self):
        correct = 0
        for headline, label in self.hashtag_headlines_labeled:
            if len(headline.split("#")) < 2 or str.count(headline, "#") > 1:
                continue
            result = sentiment_analysis(headline)
            # 0: sarcasm, 1: not sarcasm, 2: not classified in the result
            # in the dataset labels, 0 - sarcastic, 1 - not sarcastic
            if result == 0 and label == 1:
                correct += 1
            elif result == 1 and label == 0:
                correct += 1
        print(f"Correct: {correct} out of {len(self.hashtag_headlines_labeled)}")
        print(f"Whole dataset: {len(self.df)}")


tester = Tester()
tester.load_headlines()
tester.test_hashtag_on_headlines()
