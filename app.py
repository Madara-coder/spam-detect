from flask import Flask, render_template, request
import pickle  # For loading the models
import string
from nltk.corpus import stopwords
import nltk
import pandas as pd
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
ps = PorterStemmer()

def transform_text(text):
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')  # Create an additional HTML file for about page

@app.route('/contact')
def contact():
    return render_template('contact.html')  # Create an additional HTML file for contact page

@app.route('/result')
def result():
    return render_template('result.html')  # Create an additional HTML file for contact page

@app.route('/predict', methods=['POST'])
def predict():
     if request.method == 'POST':
        input_sms = request.form['sms']

        # Load the TF-IDF vectorizer and model
        tfidf= pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        model1 = pickle.load(open('model1.pkl', 'rb'))
        # Preprocess the text
        transformed_sms = transform_text(input_sms)

        # Vectorize the input
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]
        result1 =model1.predict(vector_input)[0]
        if result == 1:
            prediction = "Spam"
        else:
            prediction = "Not Spam"
        if result1 == 1:
            prediction1 = "Spam"
        else:
            prediction1 = "Not Spam"

        return render_template('result.html', prediction=prediction,prediction1 = prediction1)


if __name__ == '__main__':
    app.run(debug=True)
