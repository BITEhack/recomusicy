import pickle
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from PIL import Image
from io import BytesIO
import numpy as np
import os
import cv2
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.utils import map_emotion, extract_features
import json

app = FastAPI()

# dataframe used as database
image_database = pd.DataFrame(columns=['user_id', 'date', 'image', 'label'])
lyrics_database = pd.DataFrame(columns=['user_id', 'date', 'artist', 'track', 'genre', 'valence', 'arousal'])
text_database = pd.DataFrame(columns=['user_id', 'date', 'text', 'label'])


# load models
images_model = tf.keras.models.load_model(os.getcwd() + '/images_model.h5')
text_model: LogisticRegression = joblib.load(os.getcwd() + '/ModelTextEmotions.sav')
lyrics_model = tf.keras.models.load_model(os.getcwd() + '/lyrics_regression.h5')

# load features
input_file = open(os.getcwd() + '/features.json')
features = json.load(input_file)

with open(os.getcwd() + '/VectorizerTextEmotions.pk', 'rb') as file:
    vectorizer: TfidfVectorizer = pickle.load(file)


def read_image_file(data) -> Image.Image:
    image = Image.open(BytesIO(data))
    return image


@app.post('/store_data', tags=['recommendations'])
async def classify_num(user_id: int = None, image: Optional[UploadFile] = File(None), lyrics: Optional[str] = None,
                       text: Optional[str] = None, artist: Optional[str] = None, track: Optional[str] = None,
                       genre: Optional[str] = None) -> dict:
    global image_database, lyrics_database, text_database, images_model, text_model, vectorizer, features
    if lyrics is not None:
        pr = list(lyrics_model.predict(extract_features(lyrics, features))[0])

        valence, arousal = pr[0], pr[1]
        # update database
        lyrics_database = lyrics_database.append({'user_id': user_id, 'date': pd.to_datetime('now'), 'artist': artist,
                                                  'track': track, 'genre': genre, 'valence': valence,
                                                  'arousal': arousal}, ignore_index=True)

        return {'valence': '{}'.format(valence)}
    elif text is not None:
        X = vectorizer.transform([text])

        prediction = map_emotion(text_model.predict(X)[0])
        # update database
        # added [0] to extract prediction and text from tables (text HAS TO BE IN ARRAY TO WORK e.g. ["sad"])
        text_database = text_database.append({'user_id': user_id, 'date': pd.to_datetime('now'),
                                              'text': text[0], 'label': prediction}, ignore_index=True)
        return {'valence and arousal': prediction}
    elif image is not None:
        # get image
        image = read_image_file(await image.read())
        image = np.asarray(image).copy()

        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_NEAREST)
        image = np.array(image)

        # basic data augmentation)
        image = image / 255.0

        predictions = images_model.predict(np.array([image]))[0].argmax()

        # update database
        image_database = image_database.append({'user_id': user_id, 'date': pd.to_datetime('now'),
                                                'image': image, 'label': predictions}, ignore_index=True)

        return {"data": "{}".format(predictions)}


# @app.get('/recommend', tags=['recommendations'])
# async def recommend(user_id: int):
#     # get most relevant information
#     last_text = text_database.where('user_id' == user_id).sort_values(by=['date'], acending=False)\
#         .iat[0, text_database.get_loc('label')]

