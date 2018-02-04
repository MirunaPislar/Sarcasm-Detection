import sys
import numpy as np
import data_processing as data_proc
import extract_statistical_features as extract_feature
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics


def build_model(train_data, train_labels, test_data, test_labels, feature_set):
    if feature_set == 'pragmatic':
        pragmatic_features_names = ['tw_len_ch', 'tw_len_tok', 'avg_len', 'capitalized', 'laughter',
                                    'user_mentions', 'negations', 'affirmatives', 'interjections',
                                    'intensifiers', 'punctuation', 'emojis', 'hashtags']
        pragmatic_activation_list = \
            [[True, True, True, True, True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, False, False, False, False, True, True, True],
             [True, True, True, True, True, True, False, False, False, False, False, False, False],
             [True, True, True, True, False, False, False, False, False, False, False, False, False]]
        select_features_and_run(train_data, train_labels, test_data, test_labels, pragmatic_activation_list,
                        pragmatic_features_names, extract_feature.get_pragmatic_features)
    elif feature_set == 'sentiment':
        sentiment_features_names = ["positive emoji", "negative emoji", "neutral emoji",
                                    "emojis pos:neg", "emojis neutral:neg",
                                    "subjlexicon weaksubj", "subjlexicon strongsubj",
                                    "subjlexicon positive", "subjlexicon negative",
                                    "subjlexicon neutral", "words pos:neg", "words neutral:neg",
                                    "subjectivity strong:weak", "total sentiment words",
                                    "Vader score negative", "Vader score positive",
                                    "Vader score neutral", "Vader score compound"]
        sentiment_activation_list = \
            [[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
             # Attempt to see if the ratios really contribute to improving the predictions, or are just noise
             [True, True, True, False, False, True, True, True, True, True, False, False, False, False, True, True, True, True],
             # Attempt to see if the subjectivity lexicon contributes or just adds noise
             [True, True, True, True, True, False, False, False, False, False, False, False, False, False, True, True, True, True],
             # Attempt to see if the emoji analysis contributes or just adds noise
             [False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True],
             # Attempt to see if the Vader sentiment analysis contributes or just adds noise
             [True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False]]

        select_features_and_run(train_data, train_labels, test_data, test_labels, sentiment_activation_list,
                                sentiment_features_names, extract_feature.get_sentiment_features)
    elif feature_set == 'syntactic':
        syntactic_features_names = ['N', 'O', 'S', '^', 'Z', 'L', 'M', 'V', 'A', 'R', '!', 'D', 'P',
                                    '&', 'T', 'X', 'Y', '#', '@', '~', 'U', 'E', '$', ',', 'G']
        syntactic_activation_list = \
            [[True, True, True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True, True, True, True],
             # Check if excluding hashtags, punctuation, user mentions (anything that is not word) improves
             [True, True, True, False, True, True, True, True, True, True, False, True, True,
              False, True, True, False, False, False, False, True, True, True, False, False, False]]
        select_features_and_run(train_data, train_labels, test_data, test_labels, syntactic_activation_list,
                                syntactic_features_names, extract_feature.get_pos_tags)


def select_features_and_run(train_tweets, train_labels, test_tweets, test_labels,
                            activation_list, features_names, get_features):
    all_train_features = collect_features(train_tweets, get_features)
    all_test_features = collect_features(test_tweets, get_features)
    """
    # This is in case we want all possible combinations of features
    combinations_of_features = len(_features_names)
    activation_list = list(itertools.product([True, False], repeat=combinations_of_features))
    activation_list = activation_list[:-1]      # exclude the option when all activations are false
    """
    for activation in activation_list:
        print_features(activation, features_names)
        selected_features = [features_names[i] for i in range(0, len(features_names))
                             if activation[i] is True]
        active_train_features = []
        for one_train_features in all_train_features:
            train_features = {key: one_train_features[key] for key in selected_features}
            active_train_features.append({**train_features})
        active_test_features = []
        for one_test_features in all_test_features:
            test_features = {key: one_test_features[key] for key in selected_features}
            active_test_features.append({**test_features})
        train_features, test_features = extract_features_from_dict(active_train_features, active_test_features)
        scaled_train_features = feature_scaling(train_features)
        scaled_test_features = feature_scaling(test_features)
        linear_svc_model(scaled_train_features, train_labels, scaled_test_features, test_labels)
        print("==================================================================")
        nonlinear_svc_model(scaled_train_features, train_labels, scaled_test_features, test_labels)


def collect_features(tweets, extract_feature_per_tweet_method):
    features = []
    for t in tweets:
        this_tweet_features = extract_feature_per_tweet_method(t.split())
        features.append({**this_tweet_features})
    return features


# Method to print the features used
# @feature_options - list of True/False values; specifies which set of features is active
# @feature_names - list of names for each feature set
def print_features(feature_options, feature_names):
    print("\n==============    FEATURES    ==============")
    for name, value in zip(feature_names, feature_options):
        line_new = '{:>20}  {:>12}'.format(name, value)
        print(line_new)
    print("============================================\n")


def extract_features_from_dict(train_features, test_features):
    # Transform the list of feature-value mappings to a vector
    vector = DictVectorizer(sparse=False)
    # Learn a list of feature name -> indices mappings and transform X_train_features
    x_train_features = vector.fit_transform(train_features).tolist()
    # Just transform the X_test_features, based on the list fitted on X_train_features
    # Disadvantage: named features not encountered during fit_transform will be silently ignored.
    x_test_features = vector.transform(test_features).tolist()
    print('Size of the feature sets: train =  ', len(x_train_features[0]), ', test = ', len(x_test_features[0]))
    return x_train_features, x_test_features


def feature_scaling(features):
    scaled_features = []
    max_per_col = []
    for i in range(len(features[0])):
        maxx = max([abs(f[i]) for f in features])
        if maxx == 0.0:
            maxx = 1.0
        max_per_col.append(maxx)
    for f in features:
        scaled_features.append([float(f[i]) / float(max_per_col[i]) for i in range(len(f))])
    return scaled_features


def linear_svc_model(x_train, y_train, x_test, y_test):
    print("\n === Linear SVC MODEL with === ")
    # Generate matrix with all C
    C_range = np.outer(np.logspace(-1, 1, 3), np.array([1, 5]))
    C_range = C_range.flatten()

    parameters = {'kernel': ['linear'], 'C': C_range}
    svm_clsf = SVC()
    grid_clsf = GridSearchCV(estimator=svm_clsf, param_grid=parameters, n_jobs=1, verbose=1)
    grid_clsf.fit(x_train, y_train)
    sorted(grid_clsf.cv_results_.keys())
    classifier = grid_clsf.best_estimator_
    y_hat = classifier.predict(x_test)
    print("Report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(y_test, y_hat)))


def nonlinear_svc_model(x_train, y_train, x_test, y_test):
    print("\n === Nonlinear SVC MODEL === ")
    # Generate matrix with all gammas
    gamma_range = np.outer(np.logspace(-1, 0, 3), np.array([1, 5]))
    gamma_range = gamma_range.flatten()
    # Generate matrix with all C
    C_range = np.outer(np.logspace(-1, 1, 3), np.array([1, 5]))
    C_range = C_range.flatten()

    parameters = {'kernel': ['rbf'], 'C': C_range, 'gamma': gamma_range}
    svm_clsf = SVC()
    grid_clsf = GridSearchCV(estimator=svm_clsf, param_grid=parameters, n_jobs=1, verbose=1)
    grid_clsf.fit(x_train, y_train)
    sorted(grid_clsf.cv_results_.keys())
    classifier = grid_clsf.best_estimator_
    y_hat = classifier.predict(x_test)
    print("Report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(y_test, y_hat)))


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


if __name__ == "__main__":
    # sys.stdout = open(os.getcwd()[:os.getcwd().rfind('/')] + '/stats/pragmatic_selection.txt', 'wt')
    train_file = "train.txt"
    test_file = "test.txt"
    word_list = "word_list.txt"

    train_tokens, filtered_train_tokens, train_pos, filtered_train_pos, train_labels, \
    test_tokens, filtered_test_tokens, test_pos, filtered_test_pos, test_labels = \
        data_proc.get_clean_data(train_file, test_file, word_list)

    feature_sets = ['pragmatic', 'sentiment', 'syntactic']
    for feature_set in feature_sets:
        build_model(train_tokens, train_labels, test_tokens, test_labels, feature_set)