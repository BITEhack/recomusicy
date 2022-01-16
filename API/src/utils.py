from typing import List

import numpy as np
import pandas as pd
import spacy
import pytextrank
from pandas import DataFrame


def map_emotion(emotion: int) -> int:
    # sadness (0), joy (1), love (2), anger (3), fear (4)
    mapper = [0, 3, 2, 1, 1]
    return mapper[emotion]


def extract_features(text: str, features: List):
    # parse text through nlp
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe("textrank")
    doc = nlp(text)

    # check which features are present
    results = pd.DataFrame(columns=features)

    for phrase in doc._.phrases:
        if phrase.text in features:
            results.at[0, phrase.text] = 1

    results = results.fillna(0)

    print(results)

    return results


def make_recommendation(text_history: DataFrame, lyrics_history: DataFrame, image_history: DataFrame):
    result = [0, 0, 0, 0]
    result[text_history.sort_values(by='data', ascending=False).at[0, 'label']] += 1
    result[lyrics_history.sort_values(by='data', ascending=False).at[0, 'label']] += 1
    result[image_history.sort_values(by='data', ascending=False).at[0, 'label']] += 1

