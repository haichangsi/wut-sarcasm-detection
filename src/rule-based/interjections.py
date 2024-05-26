import nltk
from nltk.tokenize import word_tokenize
import json


nltk.download("averaged_perceptron_tagger")


def load_known_interjections(file_path, language):
    with open(file_path, "r") as file:
        data = json.load(file)
        return data.get(language, [])


def analyze_interjections(text: str, loaded_interjections: list) -> set:
    tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    interjections = [
        word
        for word, pos in tagged_tokens
        if pos == "UH" or word.lower() in loaded_interjections
    ]
    return set(interjections)


def test():
    english_interjections = load_known_interjections(
        "interjections.json", "english"
    )
    test_en_text = "Wow! That is simply unbelievable! Awesome job! Hey, are you okay? Oops. Whoa. Yikes. I did it" \
                   " again! Oh. Hey. Wow. Ouch. Hurray. Oh no. Oh well. Oh my. Ah."
    interjections = analyze_interjections(test_en_text, english_interjections)
    print(interjections)


def classify_by_interjections(text: str, loaded_interjections: list) -> bool:
    """Classifies text as sarcastic if it contains an interjection at the beginning of the text.
    Args:
            text (str): text to analyze
            loaded_interjections (list): list of interjections in the chosen language
    """
    tokens = word_tokenize(text)
    if tokens[0].lower() in loaded_interjections:
        return True

    tagged_tokens = nltk.pos_tag(tokens)
    first_tag = tagged_tokens[0][1]
    if first_tag == "UH":
        return True

    return False


test()
