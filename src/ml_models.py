import os, time, itertools
import extract_quick_and_dirty_features as extract_features
import extract_statistical_features as extract_stat_features
import utils, classifiers
import data_processing as data_proc


def baseline(tweets_train, labels_train, tweets_test, labels_test):
    # Import the subjectivity lexicon
    subj_dict = data_proc.get_subj_lexicon()

    types_of_features = ['1', '2', '3'] # , 'ngrams']
    for type in types_of_features:
        start = time.time()
        utils.print_model_title("Classification using feature type " + type)
        if type is '1':
            x_train_features = extract_features.get_features1(tweets_train, subj_dict)
            x_test_features = extract_features.get_features1(tweets_test, subj_dict)

        if type is '2':
            x_train_features = extract_features.get_features2(tweets_train, subj_dict)
            x_test_features = extract_features.get_features2(tweets_test, subj_dict)

        if type is '3':
            x_train_features = extract_features.get_features3(tweets_train, subj_dict)
            x_test_features = extract_features.get_features3(tweets_test, subj_dict)

        if type is 'ngrams':
            ngram_map, x_train_features = extract_features.get_ngram_features(tweets_train, n=1)
            x_test_features = extract_features.get_ngram_features_from_map(tweets_test, ngram_map, n=1)

        # Get the class ratio
        class_ratio = utils.get_classes_ratio_as_dict(labels_train)

        # Train on a Linear Support Vector Classifier
        print("\nEvaluating a linear SVM model...")
        classifiers.linear_svm(x_train_features, labels_train, x_test_features, labels_test, class_ratio)

        # Train on a Logistic Regression Classifier
        print("\nEvaluating a logistic regression model...")
        classifiers.logistic_regression(x_train_features, labels_train, x_test_features, labels_test, class_ratio)
        end = time.time()
        print("Completion time of the baseline model with features type %s: %.3f s = %.3f min"
              % (type, (end - start), (end - start) / 60.0))


def ml_model(train_file, test_file, train_tokens, filtered_train_tokens, train_pos,
             filtered_train_pos, train_labels, test_tokens, filtered_test_tokens, test_pos,
             filtered_test_pos, test_labels, option):
    start = time.time()

    print("Processing TRAIN SET features...\n")
    x_train_features = extract_stat_features.get_feature_set \
        (train_file, train_tokens, train_pos, filtered_train_tokens, filtered_train_pos,
         pragmatic=option[4], pos_unigrams=option[3], pos_bigrams=option[2],
         lexical=False, ngram_list=[1], sentiment=option[1], topic=option[0])

    print("Processing TEST SET features...\n")
    x_test_features = extract_stat_features.get_feature_set \
        (test_file, test_tokens, test_pos, filtered_test_tokens, filtered_test_pos,
         pragmatic=option[4], pos_unigrams=option[3], pos_bigrams=option[2],
         lexical=False, ngram_list=[1], sentiment=option[1], topic=option[0])

    train_features, test_features = utils.extract_features_from_dict(x_train_features, x_test_features)

    print("===================================================================")
    for t in test_features:
        print(t)

    # Scale the features
    scaled_train_features = utils.feature_scaling(train_features)
    scaled_test_features = utils.feature_scaling(test_features)

    # Run the models
    # utils.run_models(scaled_train_features, train_labels, scaled_test_features, test_labels)

    end = time.time()
    print("Completion time of the ML model: %.3f s = %.3f min" % ((end - start), (end - start) / 60.0))


if __name__ == "__main__":
    path = os.getcwd()[:os.getcwd().rfind('/')]
    to_write_filename = path + '/stats/8topic_models.txt'
    utils.initialize_writer(to_write_filename)

    train_file = "train_sample.txt"
    test_file = "test_sample.txt"
    word_list = "word_list.txt"

    train_tokens, filtered_train_tokens, train_pos, filtered_train_pos, train_labels, \
        test_tokens, filtered_test_tokens, test_pos, filtered_test_pos, test_labels = \
        data_proc.get_clean_data(train_file, test_file, word_list)

    run_baseline = False

    if run_baseline:
        baseline(train_tokens, train_labels, test_tokens, test_labels)
    else:
        # sets_of_features = 5
        # feature_options = list(itertools.product([False, True], repeat=sets_of_features))
        # feature_options = feature_options[1:]
        feature_options = [[True, False, False, False, False]]
        for option in feature_options:
            utils.print_features(option, ["LDA topics", "Sentiment", "POS Bigrams", "POS Unigrams", "Pragmatic"])
            start = time.time()
            ml_model(train_file, test_file, train_tokens, filtered_train_tokens, train_pos,
                     filtered_train_pos, train_labels, test_tokens, filtered_test_tokens, test_pos,
                     filtered_test_pos, test_labels, option)
