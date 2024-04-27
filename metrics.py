import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import nltk

def transform_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []

    stopwords_present = [word for word in text if word in stopwords.words("english")]

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        stemmed_word = ps.stem(i)
        y.append(stemmed_word)

    return " ".join(y)

def train_models_and_calculate_metrics(data_path):
    # Load data
    df = pd.read_csv(data_path, encoding='ISO-8859-1')
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

    # Preprocess text
    df['transformed_text'] = df['text'].apply(transform_text)

    # Split data
    X = df['transformed_text']
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize text
    vectorizer = TfidfVectorizer()
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    # Train models
    models = {
        'NB': MultinomialNB(),
        'LR': LogisticRegression(max_iter=1000),
        'DT': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC()
    }

    for name, model in models.items():
        model.fit(X_train_vect, y_train)

    # Calculate metrics
    metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_vect)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        metrics[name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

    return metrics