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

def load_classifier_model():
    
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

def get_stock_predictions_prophet(stock, days_to_get, value_to_get):
	
	now = datetime.datetime.today()
	date = now.strftime("%Y-%m-%d")
	delta = datetime.timedelta(days=120)
	start = now - delta
	start = start.strftime("%Y-%m-%d")
	data = yf.download(stock, start=start, end=date)

	data['ds'] = data.index
	data['y'] = data[value_to_get]
	df = data[['ds', 'y']]

	print("Modeling the data using Prophet")
	model = Prophet(seasonality_mode="multiplicative", daily_seasonality=True)
	model.fit(df)
	future = model.make_future_dataframe(periods=days_to_get)
	print("Making predictions")
	forecast = model.predict(future)

	return forecast[['ds', 'yhat']]

def get_stock_predictions_hwes(stock, days_to_get, value_to_get):

  now = datetime.datetime.today()
  date = now.strftime("%Y-%m-%d")
  delta = datetime.timedelta(days=45)
  start = now - delta
  start = start.strftime("%Y-%m-%d")
  data = yf.download(stock, start=start, end=date)

  data['date'] = date.index
  data.index = list(range(0, data.shape[0]))
  model = holtwinters.ExponentialSmoothing(data[value_to_get], seasonal='multiplicative', 
                              trend='multiplicative', seasonal_periods=7)
  model_fit = model.fit()
  values = model_fit.forecast(days_to_get)
  return values.values

