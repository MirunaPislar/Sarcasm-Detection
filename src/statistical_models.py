import os, time, sys, itertools
import time
import numpy as np
import extract_quick_and_dirty_features as extract_features
import extract_statistical_features as extract_stat_features
import data_processing as data_proc
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

# sys.stdout = open(os.getcwd()[:os.getcwd().rfind('/')] + '/stats/output.txt', 'wt')


def svm1(Xtrain, Ytrain, Xtest, Ytest):
    print("\n === SVC MODEL === ")
    svc = SVC()
    svc.fit(Xtrain, Ytrain)
    yhat = svc.predict(Xtest)
    print_statistics(Ytest, yhat)
    print("\n === Linear SVC MODEL === ")
    linear_svc = LinearSVC()
    linear_svc.fit(Xtrain, Ytrain)
    yhat = linear_svc.predict(Xtest)
    print_statistics(Ytest, yhat)


def svm2(Xtrain, Ytrain, Xtest, Ytest):
    C_values = [100.0, 10.0, 1.0, 0.1, 0.01, 0.001]
    for C_value in C_values:
        print("\n === Linear SVC MODEL with C = %.3f === " % C_value)
        linear_svc = LinearSVC(C=C_value, verbose=True)
        linear_svc.fit(Xtrain, Ytrain)
        yhat = linear_svc.predict(Xtest)
        print_statistics(Ytest, yhat)


def getRegularizationValues():
    values = [2 ** p for p in np.arange(0, 20, 5)] + [0.5 ** p for p in np.arange(0, 20, 5)]
    return sorted(values)


def chooseBestModel(results):
    bestResult = max(results, key=lambda x: x["f_score"])
    return bestResult


def logistic_regression(Xtrain, Ytrain, Xtest, Ytest):
    print("\n === Logistic Regression MODEL === ")
    results = []
    regValues = getRegularizationValues()
    for reg in regValues:
        print("Training for reg = %f" % reg)
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


def print_statistics(y, y_pred, printOnScreen=True):
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


def classification(type_of_features_to_use, path, tweets_train, labels_train, tweets_test, labels_test):
    # Import the subjectivity lexicon
    subj_dict = data_proc.get_subj_lexicon(path)

    print("Processing features...")

    if type_of_features_to_use == '1':
        x_train_features = extract_features.get_features1(tweets_train, subj_dict)
        x_test_features = extract_features.get_features1(tweets_test, subj_dict)

    if type_of_features_to_use == '2':
        x_train_features = extract_features.get_features2(tweets_train, subj_dict)
        x_test_features = extract_features.get_features2(tweets_test, subj_dict)

    if type_of_features_to_use == '3':
        x_train_features = extract_features.get_features3(tweets_train, subj_dict)
        x_test_features = extract_features.get_features3(tweets_test, subj_dict)

    if type_of_features_to_use == 'ngrams':
        n = 1
        ngram_map, xX_train_features = extract_features.get_ngram_features(tweets_train, n)
        x_test_features = extract_features.get_ngram_features_from_map(tweets_test, ngram_map, n)

    # Train on a Support Vector Classifier
    print("Evaluating SVM models...")
    svm1(x_train_features, labels_train, x_test_features, labels_test)

    # Train on a logistic regression model
    print("\nEvaluating Logistic Regression model...")
    logistic_regression(x_train_features, labels_train, x_test_features, labels_test)


def baseline(path, tweets_train, labels_train, tweets_test, labels_test):
    types_of_features = ['1', '2', '3', 'ngrams']
    for type in types_of_features:
        print("\n=======================================================\n")
        print("Classification using feature type \"%s\"." % type)
        print("\n=======================================================\n")
        classification(type, path, tweets_train, labels_train, tweets_test, labels_test)


def statistical_model(train_file, test_file, train_tokens, filtered_train_tokens, train_pos,
                      filtered_train_pos, train_labels, test_tokens, filtered_test_tokens,
                      test_pos, filtered_test_pos, test_labels, option):
    print("Processing TRAIN SET features...\n")
    x_train_features = extract_stat_features.get_feature_set\
        (train_file, train_tokens, train_pos, filtered_train_tokens, filtered_train_pos,
         pragmatic=option[4], pos_unigrams=option[3], pos_bigrams=option[2],
         lexical=False, ngram_list=[1], sentiment=option[1], topic=option[0])
    print("Processing TEST SET features...\n")
    x_test_features = extract_stat_features.get_feature_set\
        (test_file, test_tokens, test_pos, filtered_test_tokens, filtered_test_pos,
         pragmatic=True, pos_unigrams=option[3], pos_bigrams=option[2],
         lexical=False, ngram_list=[1], sentiment=option[1], topic=option[0])
    # Transform the list of feature-value mappings to a vector
    vector = DictVectorizer(sparse=False)
    # Learn a list of feature name -> indices mappings and transform X_train_features
    x_train_features = vector.fit_transform(x_train_features).tolist()
    # Just transform the X_test_features, based on the list fitted on X_train_features
    # Disadvantage: named features not encountered during fit_transform will be silently ignored.
    x_test_features = vector.transform(x_test_features).tolist()

    print('Size of the feature sets is', len(x_train_features[0]))

    # Train on a Support Vector Classifier
    print("\nEvaluating SVM models...")
    svm2(x_train_features, train_labels, x_test_features, test_labels)

    # Train on a Logistic Regression model
    print("\nEvaluating Logistic Regression model...")
    logistic_regression(x_train_features, train_labels, x_test_features, test_labels)


def build_models(path, train_file, test_file, train_tokens, filtered_train_tokens, train_pos, filtered_train_pos, train_labels,
                 test_tokens, filtered_test_tokens, test_pos, filtered_test_pos, test_labels, run_baseline=False):
    if run_baseline:
        start = time.time()
        baseline(path, train_tokens, train_labels, test_tokens, test_labels)
        end = time.time()
        print("Completion time of fast-baseline model: %.3f s = %.3f min" % ((end - start), (end - start) / 60.0))
    else:
        # sets_of_features = 5
        # feature_options = list(itertools.product([False, True], repeat=sets_of_features))
        # feature_options = feature_options[1:]
        feature_options=[[False, True, False, False, False]]
        for option in feature_options:
            data_proc.print_features(option, ["LDA topics", "Sentiment", "POS Bigrams", "POS Unigrams", "Pragmatic"])
            start = time.time()
            statistical_model(train_file, test_file, train_tokens, filtered_train_tokens, train_pos,
                              filtered_train_pos, train_labels, test_tokens, filtered_test_tokens, test_pos,
                              filtered_test_pos, test_labels, option)
            end = time.time()
            print("Completion time of statistical model: %.3f s = %.3f min"
                  % ((end - start), (end - start) / 60.0))


if __name__ == "__main__":
    path = os.getcwd()[:os.getcwd().rfind('/')]
    train_file = "train.txt"
    test_file = "test.txt"
    word_list = "word_list.txt"

    train_tokens, filtered_train_tokens, train_pos, filtered_train_pos, train_labels, \
    test_tokens, filtered_test_tokens, test_pos, filtered_test_pos, test_labels = \
        data_proc.get_clean_data(train_file, test_file, word_list)
    build_models(path, train_file, test_file, train_tokens, filtered_train_tokens, train_pos,
                 filtered_train_pos, train_labels, test_tokens, filtered_test_tokens,
                 test_pos, filtered_test_pos, test_labels)
