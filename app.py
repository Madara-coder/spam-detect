from flask import Flask, render_template, request
import pickle  # For loading the models
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

# Load the TF-IDF vectorizer and model
tfidf= pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
model1 = pickle.load(open("model1.pkl", "rb"))

#  Preprocesses input text by converting it to lowercase, tokenizing it, removing non-alphanumeric characters, stopwords,
#  and punctuation, and finally applying stemming to normalize words.
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
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

df['transformed_text'] = df['text'].apply(transform_text)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
from sklearn.metrics import accuracy_score
model1.fit(X_train,y_train)

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
    return render_template("result.html")  # Create an additional HTML file for contact page

@app.route("/predict", methods=["POST"])
def predict():
     if request.method == "POST":
        input_sms = request.form["sms"]

        # Preprocess the text
        transformed_sms = transform_text(input_sms)

        # Vectorize the input ?? need to ask
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

        y_pred2 = model1.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred2)
        y_pred = model.predict(X_test)
        accuracy1=accuracy_score(y_test,y_pred)
        return render_template('result.html', prediction=prediction,prediction1 = prediction1, accuracy=accuracy,accuracy1=accuracy1)


if __name__ == "__main__":
    app.run(debug=True)
