import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import gdown
import yfinance as yf
import datetime
from fbprophet import Prophet
from sklearn import model_selection, metrics
from statsmodels.tsa import holtwinters

def get_data(path):

	df = pd.read_csv(path, header = None)
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

def train_classifier_model(classifier_model, train_df, loss, opt, ep, bt, v_s):
	
	model = classifier_model
	opt = opt
	loss = loss
	model.compile(optimizer=opt,
				  loss=loss,
				  metrics='accuracy')
	
	history = model.fit(train_df['text'],
						train_df['label'],
						epochs=ep,
						batch_size=bt,
						validation_split=v_s)
	
	return model, history

def predict_classifier_model(model, sentences):
	preds = model.predict(sentences)
	
	return preds
	
def load_sentiment_model():
    
    url = 'https://drive.google.com/uc?id=1CFk7Zh6qbh6t48_eU0L1bcxnva81TQE5'
    l_path = '/tmp/model.h5'
    gdown.download(url, l_path, quiet=False)
    loaded_model = tf.keras.models.load_model(l_path, custom_objects={'KerasLayer': hub.KerasLayer})    
    return loaded_model

def predict_single_sentiment(loaded_model):
	
    input_text = input()
    preds = loaded_model.predict([input_text])    
    label_dict = {0:"neutral", 1:"negative", 2:"positive"}
    return label_dict[np.argmax(preds)], np.max(preds)

def predict_batch_sentiment(loaded_model, sentences):

    preds = loaded_model.predict(sentences)
    return sentences, preds
