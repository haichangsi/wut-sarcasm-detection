import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import words
import wordninja

nltk.download("vader_lexicon")
nltk.download("words")

sarcasm_map = {0: "sarcasm", 1: "not sarcasm", 2: "not classified"}


def extract_hashtag_words(hashtag: str, language_words: set) -> list:
    """Extracts words from a hashtag for every chosen language.
    Args:
                    hashtag (string): hashtag to extract words from
                    language_words (set): set of words for the chosen language
    """
    splits = []
    i = 0
    while i < len(hashtag):
        for j in range(len(hashtag), i, -1):
            if hashtag[i:j] in language_words:
                splits.append(hashtag[i:j])
                i = j - 1
                break
        i += 1

    return splits


def classify_by_compund(score: float) -> str:
    """Classifies sentiment based on the compound score.
    Args:
                    score (float): compound score from SentimentIntensityAnalyzer
    """
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


def sentiment_analysis(text: str) -> str:
    text, hashtag = text.split("#")
    # a simple way, works for English only - to be tested
    hashtag_words = wordninja.split(hashtag)
    hashtag = " ".join(hashtag_words)
    sia = SentimentIntensityAnalyzer()
    text_polarity = sia.polarity_scores(text)
    hashtag_polarity = sia.polarity_scores(hashtag)
    text_class = classify_by_compund(text_polarity["compound"])
    hashtag_class = classify_by_compund(hashtag_polarity["compound"])
    # 0: sarcasm, 1: not sarcasm, 2: not classified
    if text_class == "neutral" or hashtag_class == "neutral":
        return 2
    elif text_class != hashtag_class:
        return 0
    else:
        return 1


test_tweet = "I am happy. #notreallyhappy"
print(sarcasm_map[sentiment_analysis(test_tweet)])
