import pickle  # For loading the models
import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords #nltk library is being used for working with the human language data.
import nltk # NLTK -> Natural Language Toolkit
import pandas as pd # pandas is used for data manipulation and analysis
from nltk.stem.porter import PorterStemmer # PorterStemmer is used for reducing words to their root or base form.
df = pd.read_csv('spam.csv',encoding=('ISO-8859-1'))

df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
ps = PorterStemmer()
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
df = df.drop_duplicates(keep='first')
def transform1_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
 # Load the TF-IDF vectorizer and model
tfidf= pickle.load(open('vectorizer.pkl', 'rb'))
NaiveBayesmodel = pickle.load(open('NBmodel.pkl', 'rb'))
LRmodel = pickle.load(open('LRmodel.pkl', 'rb'))
DTmodel = pickle.load(open('DTmodel.pkl', 'rb'))
KNNmodel = pickle.load(open('KNNmodel.pkl', 'rb'))
SVMmodel = pickle.load(open('SvMmode.pkl', 'rb'))
df['transformed_text'] = df['text'].apply(transform1_text)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
KNNmodel.fit(X_train,y_train)
DTmodel.fit(X_train,y_train)
X_test_prediction_knn = KNNmodel.predict(X_test)
precision_knn = precision_score(y_test, X_test_prediction_knn)
recall_knn = recall_score(y_test, X_test_prediction_knn)
f1_knn = f1_score(y_test, X_test_prediction_knn)
accuracy_knn = accuracy_score( y_test, X_test_prediction_knn)


X_test_prediction_dt = DTmodel.predict(X_test)
precision_dt = precision_score(y_test, X_test_prediction_dt)
recall_dt = recall_score(y_test, X_test_prediction_dt)
f1_dt = f1_score(y_test, X_test_prediction_dt)
accuracy_dt = accuracy_score( y_test, X_test_prediction_dt)
print("TEST")
print("Precision:",precision_dt)
print("Recall:",recall_dt)
print("Precision:",f1_dt)
print("Accuracy:",accuracy_dt)
