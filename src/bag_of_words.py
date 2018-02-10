import time
import os
from pandas import DataFrame
import data_processing as data_proc
import utils
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

results = DataFrame()


def analyse_features(train_set, path, vocab_filename, use_tweet_tokenize):
    print("Analysing coefficients...")
    tweets, labels = data_proc.process_set(train_set, vocabulary_filename=vocab_filename)
    cv = CountVectorizer()
    cv.fit(tweets)
    X_train = cv.transform(tweets)
    svm = LinearSVC()
    svm.fit(X_train, labels)
    utils.plot_coefficients(svm, cv.get_feature_names(), path, top_features=5)


# Fit and evaluate SVC model
def svm(Xtrain, Ytrain, Xtest, Ytest):
    print("\n=== Linear SVC MODEL === ")
    eval_start_time = time.time()
    linear_svc = LinearSVC()
    print("Fitting Linear SVC...")
    linear_svc.fit(Xtrain, Ytrain)
    print("SVC evaluation...")
    Yhat = linear_svc.predict(Xtest)
    utils.print_statistics(Ytest, Yhat)
    eval_end_time = time.time()
    print("BoW with SVC completion time: %.3f s = %.3f min"
          % ((eval_end_time - eval_start_time),(eval_end_time - eval_start_time) / 60.0))


# Fit and evaluate feed-forward Neural Network model
def nn(x_train, y_train, x_test, y_test, path, no_of_epochs = 50, batch_size = 32, mode='tfidf',
       hl_activation_function='relu', ol_activation_function='sigmoid', save=True, plot_graph=True):
    print("\n=== Feed-forward NN model ===")
    print("List of architectural choices for this run: ")
    print("no. of epochs = %d, batch size = %d, hidden layer activation = %s, output layer activation = %s."
          % (no_of_epochs, batch_size, hl_activation_function, ol_activation_function))
    eval_start_time = time.time()
    # Build the model
    model = Sequential()
    # Use a single hidden layer with 50 neurons
    model.add(Dense(50, input_shape=(x_train.shape[1],), activation=hl_activation_function))
    # The output layer is a single neuron
    model.add(Dense(1, activation=ol_activation_function))
    model.summary()
    # Train using: binary cross entropy loss, Adam implementation of Gradient Descent
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', utils.f1_score])
    # Fit the model on the training data
    print("Fitting feed-forward NN model...")
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=no_of_epochs, verbose=2)

    if plot_graph:
        plot_name = path + "/plots/plot_using_tweet_tknzr_" + mode + "_mode_" + str(no_of_epochs) + "epochs.png"
        utils.plot_statistics(history, plot_name)

    # Evaluate the model so far
    print("NN evaluation...")
    loss, acc, f1 = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    results[mode] = [loss, acc, f1]
    classes = model.predict_classes(x_test, batch_size=batch_size)
    y_pred = [item for c in classes for item in c]
    utils.print_statistics(y_test, y_pred)
    print("No of examples predicted correctly: ", np.sum(y_test == y_pred))
    print("Accuracy, mine = %.3f, keras = %.3f." % (np.sum(y_test == y_pred) * 100.0 / y_test.size, acc * 100.0))

    if save:
        json_name = path + "/models/json_bow_nn_" + mode + "_mode_" + str(no_of_epochs) + "epochs.json"
        h5_weights_name = path + "/models/h5_bow_nn_" + mode + "_mode_" + str(no_of_epochs) + "epochs.json"
        utils.save_model(model, json_name=json_name, h5_weights_name=h5_weights_name)

    eval_end_time = time.time()
    print("Bow with NN completion time: %.3f s = %.3f min"
          % ((eval_end_time - eval_start_time), (eval_end_time - eval_start_time) / 60.0))


def bag_of_words(train_set, test_set, vocab_filename, path, word_list, mode):
    start = time.time()
    train_tweets, train_labels = data_proc.process_set(train_set, vocab_filename, word_list)
    test_tweets, test_labels = data_proc.process_set(test_set, vocab_filename, word_list)
    x_train, x_test = utils.encode_text_as_matrix(train_tweets, test_tweets, mode)
    svm(x_train, train_labels, x_test, test_labels)
    nn(x_train, train_labels, x_test, test_labels, path, mode=mode, no_of_epochs=10,
       batch_size=32, save=False, plot_graph=False)
    end = time.time()
    print("BoW model analysis completion time: %.3f s = %.3f min" % ((end - start), (end - start) / 60.0))


def main(use_tweet_tokenize=True, make_feature_analysis=True):
    path = os.getcwd()[:os.getcwd().rfind('/')]
    train_set = path + "/res/train_sample.txt"
    test_set = path + "/res/test_sample.txt"
    word_list = path + "/red/word_list.txt"

    if use_tweet_tokenize:
        vocab_filename = path + "/res/vocabulary_tok.txt"
    else:
        vocab_filename = path + "/res/vocabulary.txt"

    modes = ['binary', 'count', 'tfidf', 'freq']
    for mode in modes:
        print("\n=======================================================\n")
        print("                      Mode :  %s" % mode)
        print("\n=======================================================\n")
        bag_of_words(train_set, test_set, vocab_filename, path, word_list, mode)
    if not results.empty:
        plt.figure()
        results.boxplot()
        plt.savefig(path + "/plots/all_modes_box_plot_nn_using_tokenizer.png")
        plt.show()
    if make_feature_analysis:
        analyse_features(train_set, path, vocab_filename=vocab_filename, use_tweet_tokenize=use_tweet_tokenize)


if __name__ == "__main__":
    main()
