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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
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
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i in range(0, len(category_names)):
        targetName = []
        for j in range(0, len(pd.Series(Y_test[:,i]).value_counts())):
            targetName.append(category_names[i] + "-" + str(j))

        print(classification_report(Y_test[:, i], y_pred[:, i], target_names=targetName))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)

        parameters = {
            'vect__ngram_range': ((1, 1), (1, 2)),
            'vect__max_df': (0.5, 0.75, 1.0),
            'vect__max_features': (None, 5000, 10000),
            'tfidf__use_idf': (True, False),
            'clf__n_estimators': [50, 100, 200],
            'clf__min_samples_split': [2, 3, 4]
        }

        cv = GridSearchCV(model, param_grid=parameters)

        import datetime
        print("Start time is: ", datetime.datetime.now())
        cv.fit(X_train, Y_train[:, 0])
        print("End time is: ", datetime.datetime.now())
        y_pred = model.predict(X_test)
        classification_report(Y_test[:, 0], y_pred, target_names=["related-0","related-1","related-2"])
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()