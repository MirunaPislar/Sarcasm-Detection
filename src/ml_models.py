import os, time, itertools
from sklearn import preprocessing
import extract_baseline_features
import extract_ml_features2 as extract_features
import utils, classifiers
import data_processing as data_proc


# Settings for the up-coming ML model
pragmatic = True
lexical = True
pos_grams = True
sentiment = True
topic = True
similarity = True
pos_ngram_list = [1]
ngram_list = [1]
embedding_dim = 100
word2vec_map = utils.load_vectors(filename='glove.6B.%dd.txt' % embedding_dim)

# Set the values for the portion fo data
n_train = 3000
n_test = 500


def baseline(tweets_train, train_labels, tweets_test, test_labels):
    # Import the subjectivity lexicon
    subj_dict = data_proc.get_subj_lexicon()

    types_of_features = ['1', '2', '3', 'ngrams']
    for t in types_of_features:
        start = time.time()
        utils.print_model_title("Classification using feature type " + t)
        if t is '1':
            x_train_features = extract_baseline_features.get_features1(tweets_train, subj_dict)
            x_test_features = extract_baseline_features.get_features1(tweets_test, subj_dict)

        if t is '2':
            x_train_features = extract_baseline_features.get_features2(tweets_train, subj_dict)
            x_test_features = extract_baseline_features.get_features2(tweets_test, subj_dict)

        if t is '3':
            x_train_features = extract_baseline_features.get_features3(tweets_train, subj_dict)
            x_test_features = extract_baseline_features.get_features3(tweets_test, subj_dict)

        if t is 'ngrams':
            ngram_map, x_train_features = extract_baseline_features.get_ngram_features(tweets_train, n=1)
            x_test_features = extract_baseline_features.get_ngram_features_from_map(tweets_test, ngram_map, n=1)

        # Get the class ratio
        class_ratio = utils.get_classes_ratio_as_dict(train_labels)

        # Train on a Linear Support Vector Classifier
        print("\nEvaluating a linear SVM model...")
        classifiers.linear_svm(x_train_features, train_labels, x_test_features, test_labels, class_ratio)

        # Train on a Logistic Regression Classifier
        print("\nEvaluating a logistic regression model...")
        classifiers.logistic_regression(x_train_features, train_labels, x_test_features, test_labels, class_ratio)
        end = time.time()
        print("Completion time of the baseline model with features type %s: %.3f s = %.3f min"
              % (t, (end - start), (end - start) / 60.0))


def ml_model(train_tokens, train_pos, y_train, test_tokens, test_pos, y_test):

    print("Processing TRAIN SET features...\n")
    start = time.time()
    train_pragmatic, train_lexical, train_pos, train_sent, train_topic, train_sim = extract_features.get_feature_set\
        (train_tokens, train_pos, pragmatic=pragmatic, lexical=lexical,
         ngram_list=ngram_list, pos_grams=pos_grams, pos_ngram_list=pos_ngram_list,
         sentiment=sentiment, topic=topic, similarity=similarity, word2vec_map=word2vec_map)
    end = time.time()
    print("Completion time of extracting train models: %.3f s = %.3f min" % ((end - start), (end - start) / 60.0))

    print("Processing TEST SET features...\n")
    start = time.time()
    test_pragmatic, test_lexical, test_pos, test_sent, test_topic, test_sim = extract_features.get_feature_set \
        (test_tokens, test_pos, pragmatic=pragmatic, lexical=lexical,
         ngram_list=ngram_list, pos_grams=pos_grams, pos_ngram_list=pos_ngram_list,
         sentiment=sentiment, topic=topic, similarity=similarity, word2vec_map=word2vec_map)
    end = time.time()
    print("Completion time of extracting train models: %.3f s = %.3f min" % ((end - start), (end - start) / 60.0))

    # Get all features together
    all_train_features = [train_pragmatic, train_lexical, train_pos, train_sent, train_topic, train_sim]
    all_test_features = [test_pragmatic, test_lexical, test_pos, test_sent, test_topic, test_sim]

    # Choose your feature options: you can run on all possible combinations of features
    sets_of_features = 6
    feature_options = list(itertools.product([False, True], repeat=sets_of_features))
    feature_options = feature_options[1:]     # skip over the option in which all entries are false

    # OR Can select just the features that you want
    # From left to right, set to true if you want the feature to be active:
    # [Pragmatic, Lexical-grams, POS-grams, Sentiment, LDA topics, Similarity]
    # feature_options = [[True, True, True, True, True, True]]

    for option in feature_options:
        train_features = [{} for _ in range(len(train_tokens))]
        test_features = [{} for _ in range(len(test_tokens))]
        utils.print_features(option, ['Pragmatic', 'Lexical-grams', 'POS-grams', 'Sentiment', 'LDA topics', 'Similarity'])

        # Make a feature selection based on the current feature_option choice
        for i, o in enumerate(option):
            if o:
                for j, example in enumerate(all_train_features[i]):
                    train_features[j] = utils.merge_dicts(train_features[j], example)
                for j, example in enumerate(all_test_features[i]):
                    test_features[j] = utils.merge_dicts(test_features[j], example)

        # Vectorize and scale the features
        x_train, x_test = utils.extract_features_from_dict(train_features, test_features)
        x_train_scaled = preprocessing.scale(x_train, axis=0)
        x_test_scaled = preprocessing.scale(x_test, axis=0)

        print("Shape of the x train set (%d, %d)" % (len(x_train_scaled), len(x_train_scaled[0])))
        print("Shape of the x test set (%d, %d)" % (len(x_test_scaled), len(x_test_scaled[0])))

        # Run the model on the selection of features made
        start = time.time()
        utils.run_supervised_learning_models(x_train_scaled, y_train, x_test_scaled, y_test)
        end = time.time()
        print("Completion time of the Linear SVM model: %.3f s = %.3f min" % ((end - start), (end - start) / 60.0))


if __name__ == "__main__":
    path = os.getcwd()[:os.getcwd().rfind('/')]
    to_write_filename = path + '/stats/ml_analysis.txt'
    utils.initialize_writer(to_write_filename)

    dataset = "ghosh"      # can be "ghosh", "riloff", "sarcasmdetection" and "ptacek"
    train_tokens, train_pos, train_labels, test_tokens, test_pos, test_labels = data_proc.get_dataset(dataset)

    run_baseline = False

    if run_baseline:
        baseline(train_tokens, train_labels, test_tokens, test_labels)
    else:
        ml_model(train_tokens, train_pos, train_labels, test_tokens, test_pos, test_labels)
