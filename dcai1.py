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

