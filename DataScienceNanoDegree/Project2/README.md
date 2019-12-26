# Disaster Response Pipeline Project

### Overview:
This folder contains files for Project #2 for Udacity's Data Scientist nano degree. It uses tweets and other SMS message data posted after a disaster, and we will build 
a machine learning model that would be able to classify various attributes of a message, such as whether it's related to a weather, shelter, aid, or other categories.

### Pre-requisite
You will need Python 3.7+, and you may need to install the following libraries if it's not there already:
    'pip install pandas'
    'pip install plotly'
    
### Project contents:
   - app
        The app content provides a Python flask application that displays various statistics of the machine learning model, and also allows you to use the 
        machine learning model to classify a message
        
   - data
        This folder contains the raw data for this project. The message file contains the raw text of the message, and the category file contains the marked data for these messages.
        We will use this data to train the machine learning model.
        
   - jupyter
        This folder contains various Jupyter notebooks used to analyze the data and build the machine learning model. The commands used here are same as the python scripts used
        in the data and model folder.
        
   - model
        This folder contains the python script use to build the machine learning model

### About the data:


### Instructions to run the application:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/
