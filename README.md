# Text Classification Model - Categorizing Sentences

## Overview
This project involves building a text classification model that categorizes sentences into one of ten categories: **Education**, **Ecommerce**, **Technology**, **Healthcare**, **Entertainment**, **Finance**, **News**, **Travel**, **Sports**, and **Other**. The dataset used is noisy and contains links, emojis, and various other irrelevant elements that need to be cleaned and preprocessed before training the model.

The final model aims to achieve good accuracy while handling unlabeled and noisy data. Only the best-performing model is submitted, which is saved and can be used for further predictions.

## Problem Statement
The goal is to build a text classification model that can accurately categorize sentences into one of the following categories:
- **Education**
- **Ecommerce**
- **Technology**
- **Healthcare**
- **Entertainment**
- **Finance**
- **News**
- **Travel**
- **Sports**
- **Other**

### Dataset:
- **Size**: 100,000 rows, 1 column
- **Columns**: The dataset contains one column with the text data, which needs to be processed.
- **Preprocessing**: The data contains noise such as links, emojis, and non-English characters, all of which need to be removed during the preprocessing phase.

## Deliverables
1. Preprocessing, training, and prediction scripts (`.ipynb`).
2. Saved model file (`.pkl` or another format).
3. At least 10 test predictions in the training script.

## Steps

### 1. Importing Libraries
We begin by importing the essential libraries used for data manipulation, tokenization, lemmatization, and model training:
```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
import nltk
from sklearn.preprocessing import FunctionTransformer
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
```

### 2. Data Loading
The dataset is loaded from a CSV file:
```python
file_path = '/content/drive/MyDrive/Globussoft Assignment/Task 2/dataset.csv'
df = pd.read_csv(file_path)
```

### 3. Data Preprocessing
The dataset is preprocessed using various steps, including:
- Converting text to lowercase
- Removing HTML tags, URLs, punctuation, stopwords, emojis, and non-English characters
- Tokenizing and lemmatizing the text

We define a pipeline to perform the above steps.

### 4. Labeling Data
A keyword-based dictionary is created for each category. The model assigns labels to each sentence based on the presence of specific keywords.

```python
categories = {
    'Ecommerce': ['deal', 'order', 'cart', 'checkout', 'shipping', 'sale', 'product', 'review', 'coupon'],
    'Education': ['learn', 'study', 'course', 'class', 'teacher', 'student', 'school'],
    # Add more categories...
}

def assign_label(text):
    for category, keywords in categories.items():
        for word in keywords:
            if word in text:
                return category
    return "Other"
```

### 5. Model Training
Two machine learning models are used for training:
- **Random Forest Classifier**
- **XGBoost Classifier**

We use the `TfidfVectorizer` to transform the text data into numerical features, then train the models using the labeled data.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pickle
```

### 6. Saving the Model
After training, the best-performing model (XGBoost Classifier in this case) is saved for later use:
```python
with open(f'xgbclassifier_model.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)
```

### 7. Testing and Predictions
Once the model is trained and saved, you can load it and make predictions:
```python
test_text = ["Join the Golden Llama Treasure Hunt in Dulles!", "Clinically proven cough medicine."]
predictions = loaded_model.predict(test_text)
```

The model will predict categories for each input text.

## Example of Predictions:
1. **Text**: "Join the Golden Llama Treasure Hunt in Dulles!"
   **Predicted Category**: Travel
2. **Text**: "Clinically proven cough medicine turns sick sleep."
   **Predicted Category**: Entertainment

## Evaluation
The model is evaluated using precision, recall, and F1-score metrics. Example output for Random Forest:
```text
Results for Random Forest:
               precision    recall  f1-score   support

    Ecommerce       0.98      0.71      0.83       273
    Education       0.90      0.66      0.76       270
    ...
    accuracy                           0.75      2688
```

## Conclusion
This model provides a robust framework for categorizing noisy, unlabeled text data into predefined categories, with a focus on preprocessing and model selection.
