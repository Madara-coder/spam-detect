from flask import Flask, render_template, request,redirect,url_for,abort
import pickle  # For loading the models
import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords #nltk library is being used for working with the human language data.
import nltk # NLTK -> Natural Language Toolkit
import pandas as pd # pandas is used for data manipulation and analysis
from nltk.stem.porter import PorterStemmer # PorterStemmer is used for reducing words to their root or base form.
df = pd.read_csv('spam.csv',encoding=('ISO-8859-1'))
app = Flask(__name__)
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
ps = PorterStemmer()
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
df = df.drop_duplicates(keep='first')

# Takes input from user and sends to the model
def transform1_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Stemming section
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
Dmodel = pickle.load(open('DTmodel.pkl', 'rb'))
KNNmodel = pickle.load(open('KNNmodel.pkl', 'rb'))
SVMmodel = pickle.load(open('SvMmode.pkl', 'rb'))
df['transformed_text'] = df['text'].apply(transform1_text)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#NaiveBayesmodel.fit(X_train,y_train)
LRmodel.fit(X_train,y_train)
KNNmodel.fit(X_train,y_train)
SVMmodel.fit(X_train,y_train)

app = Flask(__name__)
ps = PorterStemmer()


#  Preprocesses input text by converting it to lowercase, tokenizing it, removing non-alphanumeric characters, stopwords.
#  and punctuation, and finally applying stemming to normalize words.
def transform_text(text):
 text_lower = text.lower()
 text_tokenize = nltk.word_tokenize(text)
 y = []

 stopwords_present = [word for word in text_tokenize if word in stopwords.words("english")]

 for i in text_tokenize:
        if i.isalnum():
            y.append(i)

 text_tokenize = y[:]
 y.clear()

 for i in text_tokenize:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

 text_tokenize = y[:]
 y.clear()

 for i in text_tokenize:
        stemmed_word = ps.stem(i)
        y.append(stemmed_word)

 return {
        "original_text": text,
        "lowercase_text": text_lower,
        "tokenized_text": text_tokenize,
        "stopwords_present": stopwords_present,
        "stemmed_text": y
    }

# Route section
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")  # Create an additional HTML file for about page

@app.route("/contact")
def contact():
    return render_template("contact.html")  # Create an additional HTML file for contact page

@app.route("/result")
def result():
    processed_text = {}
    return render_template("result.html",processed_text=processed_text)  # Create an additional HTML file for contact page

@app.route("/predict", methods=["POST"])
def predict():
     if request.method == "POST":
        input_sms = request.form["sms"]
        selected_model = request.form["model"]  # Get the selected model

        # Load the TF-IDF vectorizer and model
        if selected_model == "Naive Bayes":
            model = NaiveBayesmodel # Load Naive Bayes model
            model_name = "Naive Bayes"
        elif selected_model == "Logistic Regression":
            model = LRmodel  # Load Logistic Regression model
            model_name = "Naive Bayes"
        elif selected_model == "Decision Tree":
            model = Dmodel  # Load Logistic Regression model
            model_name = "Naive Bayes"
        elif selected_model == "KNN":
            model = KNNmodel  # Load Logistic Regression model
            model_name = "Naive Bayes"
        elif selected_model == "SVM":
            model = SVMmodel  # Load Logistic Regression model

        # Preprocess the text
        processed_text = transform_text(input_sms)
        transformed_sms = transform1_text(input_sms)

        # Vectorize the input ?? need to ask
        vector_input = tfidf.transform([transformed_sms])

        # Predict (Same as TF-IDF)
        result = model.predict(vector_input)[0]
        if result == 1:
            prediction = "Spam"
        else:
            prediction = "Not Spam"

        return render_template("result.html", prediction=prediction, processed_text=processed_text,selected_model=selected_model)

@app.route("/accuracy", methods=["GET", "POST"])
def show_accuracy():
    if request.method == 'POST':
        selected_model = request.form.get('model_select')
        return redirect(url_for('accuracy', model=selected_model))
    else:
        metrics = {}
        images = {}
        models = {
            'NB': NaiveBayesmodel,
            'LR': LRmodel,
            'DT': Dmodel,
            'KNN': KNNmodel,
            'SVM': SVMmodel
        }
        for model_name, model in models.items():
            y_test_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)
            metrics[model_name] = {
                'precision': "{:.2f}%".format(precision_score(y_test, y_test_pred) * 100),
                'recall': "{:.2f}%".format(recall_score(y_test, y_test_pred) * 100),
                'f1_score': "{:.2f}%".format(f1_score(y_test, y_test_pred) * 100),
                'accuracy': "{:.2f}%".format(accuracy_score(y_test, y_test_pred) * 100),
                'precision_train': "{:.2f}%".format(precision_score(y_train, y_train_pred) * 100),
                'recall_train': "{:.2f}%".format(recall_score(y_train, y_train_pred) * 100),
                'f1_score_train': "{:.2f}%".format(f1_score(y_train, y_train_pred) * 100),
                'accuracy_train': "{:.2f}%".format(accuracy_score(y_train, y_train_pred) * 100)
            }

        # Determine the image filename based on the model
        images[model_name] = {
            'filename': model_name.lower() + '.png'
        }
        return render_template("accuracy.html", metrics=metrics, images=images)

@app.route("/accuracy/<model>")
def accuracy(model):
    models = {
        'NB': NaiveBayesmodel,
        'LR': LRmodel,
        'DT': Dmodel,
        'KNN': KNNmodel,
        'SVM': SVMmodel
    }
    if model not in models:
        abort(404)  # Model not found, return 404 error
    else:
        model_name = model
        model = models[model_name]
        y_test_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        metrics = {
            'precision': "{:.2f}%".format(precision_score(y_test, y_test_pred) * 100),
            'recall': "{:.2f}%".format(recall_score(y_test, y_test_pred) * 100),
            'f1_score': "{:.2f}%".format(f1_score(y_test, y_test_pred) * 100),
            'accuracy': "{:.2f}%".format(accuracy_score(y_test, y_test_pred) * 100),
            'precision_train': "{:.2f}%".format(precision_score(y_train, y_train_pred) * 100),
            'recall_train': "{:.2f}%".format(recall_score(y_train, y_train_pred) * 100),
            'f1_score_train': "{:.2f}%".format(f1_score(y_train, y_train_pred) * 100),
            'accuracy_train': "{:.2f}%".format(accuracy_score(y_train, y_train_pred) * 100)
        }
        image_filename = model_name.lower() + '.png'

        return render_template("accuracy.html", metrics={model_name: metrics}, selected_model=model_name ,image_filename=image_filename)


if __name__ == "__main__":
    app.run(debug=True)
