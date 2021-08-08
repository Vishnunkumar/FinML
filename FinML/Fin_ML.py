import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
from sklearn import model_selection, metrics

def get_data(path):

    df = pd.read_csv('input/sentiment-analysis-for-financial-news/all-data.csv', header = None)
    df.columns = ['label', 'text']
    c = len(set(df['label'].values))
    
    return df, c


def classifier_model(preprocess_layer, encoder_layer, c):
    
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.KerasLayer(preprocess_layer)
    encoder_inputs = preprocessor(text_input)
    encoder = hub.KerasLayer(encoder_layer,
                             trainable=True)
    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"]
    x = tf.keras.layers.Dense(128, activation="relu")(pooled_output)
    x = tf.keras.layers.Dropout(0.5)(x)
    softmax_output = tf.keras.layers.Dense(c, activation='softmax')(x)
    model = tf.keras.Model(text_input, softmax_output)
    
    return model


def load_classifier_model(l_path):

    loaded_model = tf.keras.models.load_model(l_path, custom_objects={'KerasLayer': hub.KerasLayer})    
    return loaded_model

def predict_classes(loaded_model):
	
    input_text = input()
    preds = loaded_model.predict([input_text])
    
    label_dict = {0:"neutral", 1:"negative", 2:"positive"}
    return label_dict[np.argmax(preds)], np.max(preds)
