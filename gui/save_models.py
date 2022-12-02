import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

training_data = pd.read_csv("../data/train.csv")
bert_encoded_training_data = pd.read_csv("../data/BertEncoding.csv", header = None)

grade_categories = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
bert_data_and_labels = pd.concat([bert_encoded_training_data, training_data[grade_categories]], axis = 1, join = 'outer', ignore_index=False, sort=False)

# Trains a classifier given an initial model, training data, labels and a category
def fit_model(clf, training_data, labels, grade_category):
    X_train, X_test, y_train, y_test = train_test_split(training_data, labels[grade_category], test_size=0.2, stratify = labels[grade_category])
    clf.fit(X_train, y_train*2)
    return clf

for grade_category in grade_categories:
    svc_clf = SVC(kernel = 'rbf', C=10)
    svc_clf = fit_model(svc_clf, bert_encoded_training_data.iloc[:,:768], training_data[grade_categories], grade_category)
    with open("models/" + grade_category + 'model.pkl','wb') as f:
        pickle.dump(svc_clf,f)

