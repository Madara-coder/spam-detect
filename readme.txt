Steps involved for running the program:

0. Install the python and virtual environment.
    -> pip install virtualenv

1. Make the virtual environment
    -> python -m venv virtual_environment_name

2.  Activate the created virtual environment
    -> virtual_environment_name/bin/activate (on mac and linux)
    -> virtual_environment_name/Scripts/activate (on windows)

3. Install flask framework
    -> pip install flask (make sure to install inside the virtual machine as well)

4. Install sklearn and all for the model building Preprocess
    -> pip install scikit-learn

4. To run the server of python
    -> python -m flask run
    -> Click on the route provided

5. Install every packages present in the import section of the app.py
    -> Make sure you install in the virtual environment