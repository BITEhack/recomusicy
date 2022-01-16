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
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = FastAPI()

# dataframe used as database
image_database = pd.DataFrame(columns=['user_id', 'date', 'image', 'label'])
lyrics_database = pd.DataFrame(columns=['user_id', 'date', 'artist', 'track', 'genre', 'valence', 'arousal'])
text_database = pd.DataFrame(columns=['user_id', 'date', 'text', 'label'])


# load models
images_model = tf.keras.models.load_model(os.getcwd() + '/images_model.h5')
text_model: LogisticRegression = joblib.load('../ModelTextEmotions.sav')

with open('../VectorizerTextEmotions.pk', 'rb') as file:
    vectorizer: TfidfVectorizer = pickle.load(file)


def read_image_file(data) -> Image.Image:
    image = Image.open(BytesIO(data))
    return image


@app.post('/classification', tags=['classification'])
async def classify_num(user_id: int = None, image: Optional[UploadFile] = File(None), lyrics: Optional[str] = None,
                       text: Optional[str] = None, artist: Optional[str] = None, track: Optional[str] = None,
                       genre: Optional[str] = None) -> dict:
    global image_database, lyrics_database, text_database, images_model, text_model, vectorizer
    if lyrics is not None:
        valence, arousal = 1, 1  # TODO change to actual model
        # update database
        lyrics_database = lyrics_database.append({'user_id': user_id, 'date': pd.to_datetime('now')}, ignore_index=True)
        return {'valence': valence,
                'arousal': arousal}
    elif text is not None:
        X = vectorizer.transform(text)

        # sadness (0), joy (1), love (2), anger (3), fear (4)
        # TODO consider mapping to arousal/valence
        prediction = text_model.predict(X)
        # update database
        # added [0] to extract prediction and text from tables (text HAS TO BE IN ARRAY TO WORK e.g. ["sad"])
        text_database = text_database.append({'user_id': user_id, 'date': pd.to_datetime('now'),
                                              'text': text[0], 'label': prediction[0]}, ignore_index=True)
        return {'text': 'model to be done'}
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
                                                'image': image, 'label': predictions})

        return {"data": "{}".format(predictions)}


# classify_num(user_id=1, text=["sad boy sad"])
# print(text_database)

