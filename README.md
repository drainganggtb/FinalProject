# Sentiment Analysis of Tweets from Mongo using NLP, Machine Learning, and Deep Learning
The main focus of this project was to characterize the sentiments associated with tweets that were scraped relating to machine learning and data science. Upon creating a livestreaming program to scrape tweets and add them to MongoDB, a sentiment analysis was performed using NLTK to return the percentage of positive and negative tweets. Next, the findings were corroborated and refined through Machine Learning (SVM) and Deep Learning (BERT) methods.

Technologies and libraries used in repo: Python, Pandas, Jupyter Notebook, Google Colaboratory, Sklearn, Mongo DB, PyMongo, NLTK, Tweepy, Matplotlib, Seaborn, Transformers, DistilBERT pretrained model, Numpy, Streamlit, Explainer Dashboard

## What is contained in this repo?

This repository contains multiple secondary projects along with the **TweepywithMongoDB.ipynb** and **DeepLearning_Model.ipynb** files that are associated with the Twitter Sentiment Analysis project. The following resources have been made available in this repo:

- **Diabetes**(*diabetes.ipynb* and *diabetes-streamlit-2.py*)<-------- add links
    - The diabetes project was designed to predict the chance that an individual would be at risk for diabetes based on factors like number of pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, and age.
    - The .ipynb file cleaned the data and performed a Random Forest classifier to the tune of 88% accuracy. 
    - The **diabetes-streamlit-2.py** Python file was created with the Streamlit library to provide a simple interactive web interface made with minimal code.
        - To run this file, input the following command in terminal:
        ```streamlit run diabetes-streamlit-2.py```
    - More infomation on Streamlit [here](https://docs.streamlit.io/en/stable/)
    - From a statistical standpoint, the predictions are not likely to be extremely predictive as they are taken from only females from a specific ethnic group. 
![diabeetus](images/diabetes.png)

- **GastroGuessr -- Abalone Age predictor**(*Abalone_ML.ipynb*)
    - This Jupyter Notebook file showcases a custom supervised machine learning algorithm to predict the age of abalone specimins based on other features given.
    - The most valuable preliminary visualization is the Seaborn heatmap of correlations between Age and other measurements. 
    ![abalone](images/abalone.png)
    - After preparing the data, two types of machine learning were run on it:
        - __Multivariable Linear Regression__ through Sklearn
        - __Classification ML__ was used in the form of KNN, Decision tree, Random Forest, and SVM.
    - Random forest proved to be the best model to predict age, and it seemed important that all factors provided be considered in the prediction.

