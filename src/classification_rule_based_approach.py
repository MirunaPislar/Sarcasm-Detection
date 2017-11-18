import os
import time
import numpy as np
import pandas as pd
import extract_features
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


def load_data(filename):
    data = pd.read_csv(filename, sep="\t", header=None)
    data.columns = ["Set", "Label", "Text"]
    return np.array(data["Text"]), np.array(data["Label"])


def print_statistics(y, y_pred, printOnScreen = True):
    accuracy = metrics.accuracy_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred, average='weighted')
    recall = metrics.recall_score(y, y_pred, average='weighted')
    f_score = metrics.f1_score(y, y_pred, average='weighted')
    if printOnScreen:
        print('Accuracy:', accuracy)
        print('Precision:', precision)
        print('Recall:', recall)
        print('F_score:', f_score)
        print(metrics.classification_report(y, y_pred))
    return accuracy, precision, recall, f_score


def svm(Xtrain, Ytrain, Xtest, Ytest):
    print("\n === SVC MODEL === ")
    svc = SVC()
    svc.fit(Xtrain, Ytrain)
    Yhat = svc.predict(Xtest)
    print_statistics(Ytest, Yhat)
    print("\n === Linear SVC MODEL === ")
    linear_svc = LinearSVC()
    linear_svc.fit(Xtrain, Ytrain)
    Yhat = linear_svc.predict(Xtest)
    print_statistics(Ytest, Yhat)


def getRegularizationValues():
    values = [2 ** p for p in np.arange(0,20,5)] + [0.5 ** p for p in np.arange(0,20,5)]
    return sorted(values)


def chooseBestModel(results):
    bestResult = max(results, key=lambda x: x["f_score"])
    return bestResult


def logistic_regression(Xtrain, Ytrain, Xtest, Ytest):
    print("\n === Linear Regression MODEL === ")
    results = []
    regValues = getRegularizationValues()
    for reg in regValues:
        print("Training for reg = %f" %reg)
        clf = LogisticRegression(C=1.0 / (reg + 1e-12))
        clf.fit(Xtrain, Ytrain)
        Yhat = clf.predict(Xtest)
        accuracy, precision, recall, f_score = print_statistics(Ytest, Yhat, printOnScreen=False)
        results.append({
            "reg": reg,
            "clf": clf,
            "acc": accuracy,
            "precision": precision,
            "recall": recall,
            "f_score": f_score})
    print("Reg\t\t\tAcc\t\tP\t\tR\t\tF1")
    for result in results:
        print("%.2E\t%.3f\t%.3f\t%.3f\t%.3f" % (
            result["reg"],
            result["acc"],
            result["precision"],
            result["recall"],
            result["f_score"]))

    bestResult = chooseBestModel(results)
    print("Best regularization value: %0.2E" % bestResult["reg"])
    print("Accuracy (%%) %.3f" % (bestResult["acc"] * 100))
    print("Precision : %.3f  Recall : %.3f  F_score : %.3f\n"
          % (bestResult["precision"], bestResult["recall"], bestResult["f_score"]))


def classification(type_of_features_to_use, path, trainset_filename, testset_filename):
    start = time.time()
    print("Loading data...")
    # Process and prepare the training and testing sets
    tweets_train, labels_train = load_data(trainset_filename)
    tweets_test, labels_test = load_data(testset_filename)

    # Import the subjectivity lexicon
    subj_dict = extract_features.get_subj_lexicon(path)

    print("Processing features...")

    if type_of_features_to_use == '1':
        X_train_features = extract_features.get_features1(tweets_train, subj_dict)
        X_test_features = extract_features.get_features1(tweets_test, subj_dict)
    if type_of_features_to_use == '2':
        X_train_features = extract_features.get_features2(tweets_train, subj_dict)
        X_test_features = extract_features.get_features2(tweets_test, subj_dict)
    if type_of_features_to_use == '3':
        X_train_features = extract_features.get_features3(tweets_train, subj_dict)
        X_test_features = extract_features.get_features3(tweets_test, subj_dict)
    if type_of_features_to_use == 'ngrams':
        n = 1
        ngram_map, X_train_features = extract_features.get_ngram_features(tweets_train, n)
        X_test_features = extract_features.get_ngram_features_from_map(tweets_test, ngram_map, n)

    # Train on a Support Vector Classifier
    print("Evaluating SVM models...")
    svm(X_train_features, labels_train, X_test_features, labels_test)

    # Train on a logistic regression model
    print("\nEvaluating Linear Regression model...")
    logistic_regression(X_train_features, labels_train, X_test_features, labels_test)

    end = time.time()
    print("Completion time: %.3f s = %.3f min" %((end - start), (end - start) / 60.0))

path = os.getcwd()[:os.getcwd().rfind('/')]
train_file = path + "/res/train.txt"
test_file = path + "/res/test.txt"
types_of_features = ['1', '2', '3', 'ngrams']
for type in types_of_features:
    print("\n=======================================================\n")
    print("Classification using feature type \"%s\"." % type)
    print("\n=======================================================\n")
    classification(type, path, train_file, test_file)
