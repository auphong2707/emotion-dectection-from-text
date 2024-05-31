# Emotion dectection from text
The project aims to create a **Machine Learning** powered model which is capable of classifying the emotions of the text (in the form of short quotes/comments) with predefined labels of emotions.

Emotion detection, also known as sentiment analysis, is now one of the most attractive subfields of Machine Learning, especially Natural Language Processing (NLP) due to its wide applications in many aspects of modern life, such as providing emotional information to help people developing insights or making decisions. In this project, we specifically try to convey the main emotion of a comment sentence. Generally, we expect the model to be able to take the input sentence in the form of short comments and returns its major emotion(s), corresponding to one of the predefined labels.

Our approach to solving this problem is to split it into two halves:

  1.   Data Preprocessing
  2.   Model training and Result

For detail of each half, please go into the section

The dataset we use in this project is obtained from Kaggle: [EDFT Dataset](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)

# Workflow
Our team developed a workflow to streamline the project process. Below is our workflow diagram:

![Workflow](./data/ML%20flowchart.jpg)


# Notebook

**For testing purpose, you can rerun our notebooks as following order**

Before testing, please remember to install requirements by using this command
```bash
pip install -r requirement.txt
```

## Data Preprocess
The notebook we use to preprocess the data is *DataPreprocess.ipynb*. This notebook demonstrates our preprocessing steps using the **sklearn** module. 

We have acomplished preprocess phase by 2 tokenization method (Bag of Words and TF-IDF) and 2 filter ways (Normal way and L1 regularization way). he preprocessed data is then exported to the */data/dataset/processed* directory

For a detailed explanation, please refer to the notebook.

## Tuning
After completing the data preprocessing phase, the next task is to tune each type of preprocessing method.

To perform this task in parallel, we created separate branches and notebooks for tuning each algorithm:
  * *K-nearest neighbors (KNN).ipynb*
  * *Naive Bayes.ipynb*
  * *Decision Tree.ipynb*
  * *Support Vector Machine (SVM).ipynb*
  * *Logistic regression (OvR).ipynb*
  * *Random Forest.ipynb*
  * *Softmax regression.ipynb*

After tuning a model by using a dataset, we export a tuning report that includes the optimized hyperparameters, which will be used in the final report and model export phases. **Then, we will use that notebook for tuning next type of preprocessing method**.

For a detailed explanation, please refer to the notebook and the reports in */data/tuning-reports*.

## Practical test
The notebook *PracticalTest.ipynb* is used to implement a practical test by gathering all the tuned hyperparameters, retraining the models with the entire original dataset, and testing the models with a separate dataset.

The test dataset is sourced from the Google Research blog: [Practical Test dataset](https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/)

## Export models

The notebook *Export Model.ipynb* used to export all the trained model pipeline. This process involves gathering all the tuned hyperparameters, retraining the models, and then exporting the pipelines to the */data/models* directory.

# User Interface

We create an user interface for anyone want to take a test to our models.

You can open our UI by running the *UI.py* using command
```bash
python UI.py
```