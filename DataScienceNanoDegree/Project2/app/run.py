import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterData', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Visual 1: Number of categories per message.
    df["numCategory"] = np.sum(df[[item for item in df.columns if item not in ['id', 'message', 'original', 'genre']]], axis=1)
    numCatHist = pd.DataFrame(df["numCategory"].value_counts())
    numCatHist = numCatHist.reset_index()
    numCatHist.columns = ["Total Category Points", "Message Count"]
    numCatHist = numCatHist.sort_values("Total Category Points")

    # Visual 2: Number of categories available
    cat_available = np.sum(df[[item for item in df.columns if item not in ['numCategory', 'id', 'message', 'original', 'genre']]]).sort_values(ascending=False)

    graphs = [
        {
            'data': [
                Bar(
                    x=numCatHist["Total Category Points"],
                    y=numCatHist["Message Count"]
                )
            ],

            'layout': {
                'title': 'Histogram of message category count',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "# of categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cat_available.index,
                    y=cat_available
                )
            ],

            'layout': {
                'title': 'Available message count by category',
                'yaxis': {
                    'title': "Message count"
                }
            }
        },

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()