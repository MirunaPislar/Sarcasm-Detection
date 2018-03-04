import time, os, utils
from pandas import DataFrame
import data_processing as data_proc
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


def rule_based(x_train, y_train, x_test, y_test, vocab_filename):
    # Build a vocabulary and count the sarcastic or non-sarcastic context in which a word appears
    vocab = data_proc.build_vocabulary(vocab_filename, x_train, minimum_occurrence=10)
    # vocab = set(' '.join([x.lower() for x in x_train]).split())
    counts = {k: [0, 0] for k in vocab}
    for tw, y in zip(x_train, y_train):
        for word in tw.split():
            word = word.lower()
            if word in vocab:
                if y == 0:
                    counts[word][0] += 1
                else:
                    counts[word][1] += 1

    # Calculate the relative weight of each word, based on the sarcastic/non-sarcastic tweets that it appears
    weight = dict.fromkeys([k for k in counts.keys()], 0)
    for word in counts.keys():
        if counts[word][1] + counts[word][0] != 0:
            weight[word] = (counts[word][1] - counts[word][0]) / (counts[word][1] + counts[word][0])

    # Rule-based predictions based on the previously calculated weigths
    y_pred = []
    for tw, y in zip(x_test, y_test):
        score = 0.0
        for word in tw.split():
            word = word.lower()
            if word in vocab:
                score += weight[word]
        if score >= 0.0:
            y_pred.append(1)
        else:
            y_pred.append(0)
    utils.print_statistics(y_test, y_pred)


# Fit and evaluate feed-forward Neural Network model
def nn_bow_model(x_train, y_train, x_test, y_test, results, mode, epochs=15, batch_size=32, hidden_units=50,
                 save=False, plot_graph=False):
    # Build the model
    print("\nBuilding Bow NN model...")
    model = Sequential()
    model.add(Dense(hidden_units, input_shape=(x_train.shape[1],), activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    # Train using: binary cross entropy loss, Adam implementation of Gradient Descent
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', utils.f1_score])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    if plot_graph:
        utils.plot_training_statistics(history, "/plots/bow_models/bow_%s_mode" % mode)

    # Evaluate the model
    loss, acc, f1 = model.evaluate(x_test, y_test, batch_size=batch_size)
    results[mode] = [loss, acc, f1]
    classes = model.predict_classes(x_test, batch_size=batch_size)
    y_pred = [item for c in classes for item in c]
    utils.print_statistics(y_test, y_pred)
    print("%d examples predicted correctly." % np.sum(np.array(y_test) == np.array(y_pred)))
    print("%d examples predicted 1." % np.sum(1 == np.array(y_pred)))
    print("%d examples predicted 0." % np.sum(0 == np.array(y_pred)))

    if save:
        json_name = path + "/models/bow_models/json_bow_" + mode + "_mode.json"
        h5_weights_name = path + "/models/bow_models/h5_bow_" + mode + "_mode.json"
        utils.save_model(model, json_name=json_name, h5_weights_name=h5_weights_name)


if __name__ == "__main__":
    path = os.getcwd()[:os.getcwd().rfind('/')]
    to_write_filename = path + '/stats/bag_of_words_analysis.txt'
    utils.initialize_writer(to_write_filename)

    train_filename = "train.txt"
    test_filename = "test.txt"
    vocab_filename = path + "/res/vocabulary/vocabulary.txt"

    # Load the data
    train_tweets = utils.load_file(path + "/res/tokens/tokens_clean_original_" + train_filename).split("\n")
    test_tweets = utils.load_file(path + "/res/tokens/tokens_clean_original_" + test_filename).split("\n")

    # Make sure all words are lower-case
    x_train = [t.lower() for t in train_tweets]
    x_test = [t.lower() for t in test_tweets]

    # Filtered based on vocabulary
    train_tweets = data_proc.filter_based_on_vocab(train_tweets, vocab_filename, min_occ=10)
    test_tweets = data_proc.filter_based_on_vocab(test_tweets, vocab_filename, min_occ=10)

    # Load the labels
    y_train = [int(l) for l in utils.load_file(path + "/res/data/labels_" + train_filename).split("\n")]
    y_test = [int(l) for l in utils.load_file(path + "/res/data/labels_" + test_filename).split("\n")]

    # A rule-based approach used here to measure and compare what is the BoW actually learning from the words
    utils.print_model_title("Rule-based approach")
    rule_based(train_tweets, y_train, test_tweets, y_test, vocab_filename)

    modes = ['binary', 'count', 'tfidf', 'freq']
    results = DataFrame()

    for mode in modes:
        utils.print_model_title("BoW Analysis for Mode %s" % mode)
        #  Encode train and test data based on the currently selected mode
        tokenizer, x_train, x_test = utils.encode_text_as_matrix(train_tweets, test_tweets, mode, lower=True)
        word_to_indices = tokenizer.word_index
        index_to_word = {i: w for w, i in word_to_indices.items()}
        start = time.time()
        utils.run_supervised_learning_models(x_train, y_train, x_test, y_test, make_feature_analysis=True,
                                             feature_names=index_to_word, top_features=20,
                                             plot_name="/bow_models/bow_%s_" % mode)
        nn_bow_model(x_train, y_train, x_test, y_test, results, mode, save=False, plot_graph=True)
        end = time.time()
        print("BoW for %s mode completion time: %.3f s = %.3f min" % (mode, (end - start), (end - start) / 60.0))

    # Plot the Bow-NN results obtained for each mode
    if not results.empty:
        plt.figure()
        results.boxplot()
        plt.savefig(path + "/plots/bow_models/bow_boxplot.png")
        plt.show()
