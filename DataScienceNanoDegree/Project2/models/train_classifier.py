import sys
import pandas as pd
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from functools import partial
from sklearn.externals import joblib
import nltk

nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("DisasterData", con=engine)
    X = df.message.values
    y_df = df[[item for item in df.columns if item not in ['id', 'message', 'original', 'genre']]]
    y_columns = y_df.columns
    y = y_df.values
    return X, y, y_columns

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=partial(tokenize))),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=20)))
    ])

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i in range(0, len(category_names)):
        print("Report for column " + category_names[i] + "\n" + classification_report(Y_test[:, i], y_pred[:, i]))
        print("\n")


def save_model(model, model_filepath):
    # pickle.dump(model, open(model_filepath, 'wb'))
    joblib.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)

        '''
        These were the original parameters used to run GridSearch and it takes forever. Picking only the best estimates 
        from the initial run. 
        parameters = {
            'vect__ngram_range': ((1, 1), (1, 2)),
            'vect__max_df': (0.5, 0.75, 1.0),
            'vect__max_features': (None, 5000, 10000),
            'tfidf__use_idf': (True, False),
            'clf__n_estimators': [50, 100, 200],
            'clf__min_samples_split': [2, 3, 4]
        }
        '''

        parameters = {
            'vect__ngram_range': [(1, 2)],
            'vect__max_df': [0.5],
            'vect__max_features': [5000],
            'tfidf__use_idf': (True, False)
        }

        cv = GridSearchCV(model, param_grid=parameters)

        import datetime
        print("Start time is: ", datetime.datetime.now())
        cv.fit(X_train, Y_train)
        print("End time is: ", datetime.datetime.now())

        print(cv.best_estimator_)
        
        print('Evaluating model...')
        evaluate_model(cv, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(cv, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()