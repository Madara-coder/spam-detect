Steps involved for running the program:

0. Install the python and virtual environment.
    -> pip install virtualenv

1. Make the virtual environment
    -> python -m venv virtual_environment_name

2.  Activate the created virtual environment
    -> virtual_environment_name/bin/activate (on mac and linux)
    -> virtual_environment_name/Scripts/activate (on windows)

Note: If permission error then enter: chmod +x virtual_environment_name/bin/activate

3. Install flask framework
    -> pip install flask (make sure to install inside the virtual machine as well)

4. Install sklearn and all for the model building Preprocess
    -> pip install scikit-learn

4. To run the server of python
    -> python -m flask run
    -> Click on the route provided

5. Install every packages present in the import section of the app.py
    -> Make sure you install in the virtual environment


// TF-IDF (Not Algorithm just used for taking alphanumeric data into numeric form)
    -> TF (Term Frequency) It is to find out how many times the word is being repeated in single statement.
    -> Idf (Inverse Document Frequency) -> sabai documentation ko collection ma tyo word kati choti aako cha bhnera calculate garni
    -> Mutiplication of both Tf and Idf is vectorizer

// ALGORITHMS:
// SVM (Complex so made using library) (Support Vector Machine)
    -> tei hyperlane ho left ra right tira bata chuttauni
    ->predict: hyperlane

// Naive Bayes
    -> Yes yes wala tyo astina padheko wala
    -> probabilities kati cha bhanera herera nikalcha
    -> Highest jun ko aauncha tei probability anusaar final result nikalcha.
    -> predict: probability

// KNN (K-Nearest Neighbour)
    -> Tyo distance wala sabai ko distance nikalera ani answer thaha huncha ani arko chahin midpoint nikalera herni
    -> prediction: distance

// Logistic Regression
    -> sigmoid : smap 1 , not-spam 0
    -> threshold: 0.5 deko huncha tyo chahin mid-point huncha which is given in model Training
    -> If the received value smaller than threshold not-spam else it will be spam data.
    -> prediction: sigmoid

// DecisionTree
    -> Simple tree is made using different nodes i.e. left and right nodes
    -> Gini impurity -> It is calculated for each word provided using formula, the minimum value of each
    gini-impurity is made child node and again the process is repeated.
    -> prediction: gini-impurity
