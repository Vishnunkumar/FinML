# Fin_ML
Deep learning applications on the financial and banking world

## Quickstart

This model is trained on financial headlines dataset. I have used "small_bert_en_uncased" for training the dataset. 
```
from Fin_ML import Fin_ML
loaded_model = Fin_ML.load_classifier_model()

# Input is called inside the function, 
prediction = Fin_ML.predict_classes(loaded_model)

Eg : Investors beware — vaccines aren’t a silver bullet for markets
prediction
prints -- ('negative', 0.9195326)
```

P.S : I will add comparisons with other sentiment analysis tool soon.

## Coming soon

- Training functionality for classification models
- Exploring on NER, text generation and will be adding soon in the future
