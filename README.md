# Fin_ML
Deep learning applications on the financial and banking world

## Installation
```
pip insall Fin-ML
```

## Dependencies
- [tensorflow](https://www.tensorflow.org/)
- [tensorflow_hub](https://tfhub.dev/)
- [pandas](https://pandas.pydata.org/)
- [sklearn](https://scikit-learn.org/)
- [numpy](https://numpy.org/)
- [gdown](https://github.com/wkentaro/gdown)
- [yfinance](https://pypi.org/project/yfinance/)
- [fbprophet](https://facebook.github.io/prophet/)
- [statsmodels](https://www.statsmodels.org/stable/index.html)

## Quickstart

This model is trained on financial headlines dataset. I have used "small_bert_en_uncased" for training the dataset. 
```
from Fin_ML import fin_nlp, stocks_forecast
loaded_model = fin_nlp.load_sentiment_model()

# Input is called inside the function, 
prediction = fin_nlp.predict_single_sentiment(model)

Eg : Investors beware — vaccines aren’t a silver bullet for markets
prediction
prints -- ('negative', 0.9195326)

# Stocks forecasting using Exponential Smoothing ("stock", "days to forecast", "Values - ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']")
stocks = stocks_forecast.exp_smoothing_forecast("TSLA", 8, "Open")

# Stocks forecasting using Prophet ("stock", "days to forecast", "Values - ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']")
stocks = stocks_forecast.prophet_forecast("TSLA", 8, "Open")

# Training pipeline
df, c = fin_nlp.get_data('/content/train.csv') - # make sure the first column is label and the second one is the text and also it must have only two columns

# creating model
prp = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
enc = "https://tfhub.dev/tensorflow/mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT/1"
model = fin_nlp.classifier_model(prp, enc, c-1, tf.keras.activations.sigmoid) - # c = number of classes, (c-1) only if its a binary classification task

model, history= fin_nlp.train_classifier_model(model, 
                                               train_df, 
                                               tf.keras.losses.BinaryCrossentropy(), 
                                               tf.keras.optimizers.Adam(lr=1e-4), 
                                               10, 
                                               32, 
                                               0.2) - # (model, train_df, loss_function, optimizer, epochs, batch_size, validation_split)

predictions = fin_nlp.predict_classifier_model(model, texts) - # (model, list of sentences)
```

### Benchmark with other sentiment tools
I have taken a sample of 100 sentences involving financial terms and analyzed it with the various sentiment analysis tools in the market and below are the observations. Due to class imbalance in the dataset I have chose F1-score for benchmarking the tools

| Sentiment analysis tool | time take for inference | F1-score |
| ------------- | ------------- | ------------- |
| Fin-ML  | 363 ms | 0.82 |
| Vader-sentiment  | 17.6 ms  | 0.26 |
| transformers-pipeline  | 4910 ms | 0.30 |

P.S : I will be adding more tools for benchmarking 

## Coming soon

- Training functionality for classification models
- Exploring on NER, text generation and will be adding soon in the future
