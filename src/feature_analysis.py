import sys, os
import numpy as np
import data_processing as data_proc
import extract_statistical_features as extract_feature
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics


def perform_feature_analysis(train_tokens, filtered_train_tokens, train_pos, filtered_train_pos, train_labels,
                             test_tokens, filtered_test_tokens, test_pos, filtered_test_pos, test_labels,
                             feature_set, features_names, activation_list, feature_function):

    # Get all the train and test features, each in their own dictionary of feature names:feature values
    all_train_features = collect_features(train_tokens, filtered_train_tokens, train_pos, filtered_train_pos,
                                          feature_set, feature_function)
    all_test_features = collect_features(test_tokens, filtered_test_tokens, test_pos, filtered_test_pos,
                                          feature_set, feature_function)
    """
    # This is in case we want to run over all possible combinations of features, 
    # and not using the activation_list argument (not done here as it is obv huge to run through all 2^|features|)
    combinations_of_features = len(_features_names)
    activation_list = list(itertools.product([True, False], repeat=combinations_of_features))
    activation_list = activation_list[:-1]      # exclude the option when all activations are false
    """
    for activation in activation_list:
        # Print the current grid of selected features
        print_features(activation, features_names)

        # Select the active features, used in current analysis
        selected_features = [features_names[i] for i in range(0, len(features_names))
                             if activation[i] is True]
        active_train_features = select_active_features(all_train_features, selected_features)
        active_test_features = select_active_features(all_test_features, selected_features)

        # Convert feature dictionary to a list of feature values (a dict vectorizer)
        train_features, test_features = extract_features_from_dict(active_train_features, active_test_features)

        # Scale the features
        scaled_train_features = feature_scaling(train_features)
        scaled_test_features = feature_scaling(test_features)

        # Run the models
        run_models(scaled_train_features, train_labels, scaled_test_features, test_labels)


# Method to print the features used
# @feature_options - list of True/False values; specifies which set of features is active
# @feature_names - list of names for each feature set
def print_features(feature_options, feature_names):
    print("\n=========================    FEATURES    =========================\n")
    for name, value in zip(feature_names, feature_options):
        line_new = '{:>30}  {:>10}'.format(name, value)
        print(line_new)
    print("\n==================================================================\n")


def select_active_features(all_features, selected_features):
        # Filter the features to select just the active ones, based on activation
        active_features = []
        for feature in all_features:
            features = {key: feature[key] for key in selected_features}
            active_features.append({**features})
        return active_features


def perform_function(passed_function, *args):
    return passed_function(*args)


def collect_features(tokens, filtered_tokens, pos_tags, filtered_pos_tags,
                     feature_set, feature_function):
    features = []
    for token, pos in zip(tokens, pos_tags):
        this_tweet_features = []
        if feature_set == 'pragmatic':
            this_tweet_features = perform_function(feature_function, token.split())
        elif feature_set == 'sentiment':
            # Emoji lexicon - underlying sentiment (pos, neutral, neg)
            emoji_dict = data_proc.build_emoji_sentiment_dictionary()
            # Obtain subjectivity features from the MPQA lexicon and build the subjectivity lexicon
            subj_dict = data_proc.get_subj_lexicon()
            this_tweet_features = perform_function(feature_function, token, token.split(),
                                                          pos.split(), emoji_dict, subj_dict)
        elif feature_set == 'syntactic':
            this_tweet_features = perform_function(feature_function, pos.split())
        features.append({**this_tweet_features})
    return features


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


def run_models(train_features, train_labels, test_features, test_labels):
    print("\n==================================================================\n")
    linear_svc(train_features, train_labels, test_features, test_labels)
    print("\n==================================================================\n")
    nonlinear_svc(train_features, train_labels, test_features, test_labels)


def linear_svc(x_train, y_train, x_test, y_test):
    print("\t\t\t\tLinear SVM")
    print("==================================================================\n")
    # Generate matrix with all gammas
    gamma_range = np.outer(np.logspace(-1, 0, 3), np.array([1, 5]))
    gamma_range = gamma_range.flatten()
    # Generate matrix with all C
    C_range = np.outer(np.logspace(-1, 1, 3), np.array([1, 5]))
    C_range = C_range.flatten()

    parameters = {'kernel': ['linear'], 'C': C_range, 'gamma': gamma_range}
    svm_clsf = SVC()
    grid_clsf = GridSearchCV(estimator=svm_clsf, param_grid=parameters, n_jobs=1, verbose=1)
    grid_clsf.fit(x_train, y_train)
    sorted(grid_clsf.cv_results_.keys())
    classifier = grid_clsf.best_estimator_
    y_hat = classifier.predict(x_test)
    print("Report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(y_test, y_hat)))


def nonlinear_svc(x_train, y_train, x_test, y_test):
    print("\t\t\t\tNonlinear SVM")
    print("==================================================================\n")
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


def build_model(train_tokens, filtered_train_tokens, train_pos, filtered_train_pos, train_labels,
                test_tokens, filtered_test_tokens, test_pos, filtered_test_pos, test_labels,
                feature_set):
    features_names = ""
    activation_list = [[]]
    feature_function = None
    if feature_set == 'pragmatic':
        features_names = ['tw_len_ch', 'tw_len_tok', 'avg_len', 'capitalized', 'laughter',
                          'user_mentions', 'negations', 'affirmatives', 'interjections',
                          'intensifiers', 'punctuation', 'emojis', 'hashtags']
        activation_list = \
            [[True, True, True, True, True, True, True, True, True, True, True, True, True],
             [True, True, True, True, True, True, False, False, False, False, True, True, True],
             [True, True, True, True, True, True, False, False, False, False, False, False, False],
             [True, True, True, True, False, False, False, False, False, False, False, False, False]]

        feature_function = extract_feature.get_pragmatic_features
    elif feature_set == 'sentiment':
        features_names = ["positive emoji", "negative emoji", "neutral emoji", "emojis pos:neg", "emojis neutral:neg",
                          "subjlexicon weaksubj", "subjlexicon strongsubj",
                          "subjlexicon positive", "subjlexicon negative", "subjlexicon neutral",
                          "words pos:neg", "words neutral:neg", "subjectivity strong:weak", "total sentiment words",
                          "Vader score neg", "Vader score pos", "Vader score neu", "Vader score compound"]
        activation_list = \
            [[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
             # Attempt to see if the ratios really contribute to improving the predictions, or are just noise
             [True, True, True, False, False, True, True, True, True, True, False, False, False, False, True, True, True, True],
             # Attempt to see if the subjectivity lexicon contributes or just adds noise
             [True, True, True, True, True, False, False, False, False, False, False, False, False, False, True, True, True, True],
             # Attempt to see if the emoji analysis contributes or just adds noise
             [False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True],
             # Attempt to see if the Vader sentiment analysis contributes or just adds noise
             [True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False]]
        feature_function = extract_feature.get_sentiment_features
    elif feature_set == 'syntactic':
        features_names = ['N', 'O', 'S', '^', 'Z', 'L', 'M', 'V', 'A', 'R', '!', 'D', 'P',
                          '&', 'T', 'X', 'Y', '#', '@', '~', 'U', 'E', '$', ',', 'G']
        activation_list = \
            [[True, True, True, True, True, True, True, True, True, True, True, True, True,
              True, True, True, True, True, True, True, True, True, True, True, True],
             # Check if excluding hashtags, punctuation, user mentions (anything that is not word) improves
             [True, True, True, False, True, True, True, True, True, True, False, True, True,
              False, True, True, False, False, False, False, True, True, True, False, False, False]]
        feature_function = extract_feature.get_pos_features

    # Perform the feature analysis - extract the active features and run the classifiers to gather conclusive results
    perform_feature_analysis(train_tokens, filtered_train_tokens, train_pos, filtered_train_pos, train_labels,
                             test_tokens, filtered_test_tokens, test_pos, filtered_test_pos, test_labels,
                             feature_set, features_names, activation_list, feature_function)


if __name__ == "__main__":
    sys.stdout = open(os.getcwd()[:os.getcwd().rfind('/')] + '/stats/pragmatic_feature_analysis.txt', 'wt')
    train_file = "train.txt"
    test_file = "test.txt"
    word_list = "word_list.txt"

    train_tokens, filtered_train_tokens, train_pos, filtered_train_pos, train_labels, \
    test_tokens, filtered_test_tokens, test_pos, filtered_test_pos, test_labels = \
        data_proc.get_clean_data(train_file, test_file, word_list)

    feature_sets = ['pragmatic']#, 'sentiment', 'syntactic']
    for feature_set in feature_sets:
        build_model(train_tokens, filtered_train_tokens, train_pos, filtered_train_pos, train_labels,
                    test_tokens, filtered_test_tokens, test_pos, filtered_test_pos, test_labels, feature_set)