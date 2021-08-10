# Fin_ML
Deep learning applications on the financial and banking world

## Quickstart

This model is trained on financial headlines dataset. I have used "small_bert_en_uncased" for training the dataset. 
```
from Fin_ML import fin_nlp, stocks_forecast
loaded_model = fin_nlp.load_classifier_model()

# Input is called inside the function, 
prediction = fin_nlp.predict_single_sentiment(model)

Eg : Investors beware — vaccines aren’t a silver bullet for markets
prediction
prints -- ('negative', 0.9195326)

# Stocks forecasting using Exponential Smoothing ("stock", "days to forecast", "Values - ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']")
stocks = stocks_forecast.exp_smoothing_forecast("TSLA", 8, "Open")

# Stocks forecasting using Prophet ("stock", "days to forecast", "Values - ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']")
stocks = stocks_forecast.prophet_forecast("TSLA", 8, "Open")
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
