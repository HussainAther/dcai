import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics

sgd_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

_ = sgd_clf.fit(train['review'], train['label'])

train = pd.read_csv('reviews_train.csv')
test = pd.read_csv('reviews_test.csv')

test.sample(5)

def evaluate(clf):
    pred = clf.predict(test['review'])
    acc = metrics.accuracy_score(test['label'], pred)
    print(f'Accuracy: {100*acc:.1f}%')

evaluate(sgd_clf)

"""
Can you train a more accurate model on the dataset (without changing the dataset)? You might find this scikit-learn classifier comparison handy, as well as the documentation for supervised learning in scikit-learn.

One idea for a model you could try is a naive Bayes classifier.

You could also try experimenting with different values of the model hyperparameters, perhaps tuning them via a grid search.

Or you can even try training multiple different models and ensembling their predictions, a strategy often used to win prediction competitions like Kaggle.
"""
