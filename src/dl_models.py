import os, sys, time, datetime
import numpy as np
import data_processing as data_proc
import utils
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, LSTM, GRU, Bidirectional
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import plot_model


# A standard DNN used as a baseline
def build_standard_dnn_model(input_shape, hidden_units=256):
    print("Standard DNN")

    model = Sequential()
    model.add(Dense(hidden_units, input_shape=(input_shape,), kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model


# A model using just convolutional neural networks
def build_conv_model(max_length, vocab_size, embedding_vector_dim=256, hidden_units=256, dropout=0.3):
    print("CONV Model")

    model = Sequential()

    # Add an embedding layer to obtain a dense vector representation of the data
    # Using Glorot normal initializer (also called Xavier normal initializer)
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_vector_dim, input_length=max_length,
                        embeddings_initializer='glorot_normal'))
    # model.add(Dropout(dropout))

    # Convolution layers
    model.add(Conv1D(filters=hidden_units, kernel_size=3, kernel_initializer='he_normal', padding='valid',
                            activation='relu', input_shape=(1, max_length)))
    # model.add(MaxPooling1D(pool_size=3))
    model.add(Conv1D(filters=hidden_units, kernel_size=3, kernel_initializer='he_normal', padding='valid',
                            activation='relu', input_shape=(1, max_length)))
    model.add(GlobalMaxPooling1D())
    # model.add(MaxPooling1D(pool_size=3))
    # model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    return model


# A model using just Long Short Term Memory units
def build_lstm_model(max_length, vocab_size, embedding_vector_dim=256, hidden_units=256, dropout=0.3):
    print("LSTM Model")

    model = Sequential()

    # Add an embedding layer to obtain a dense vector representation of the data
    # Using Glorot normal initializer (also called Xavier normal initializer)
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_vector_dim, input_length=max_length,
                        embeddings_initializer='glorot_normal'))
    # LSTM Layers
    model.add(LSTM(hidden_units, kernel_initializer='he_normal', activation='tanh', dropout=dropout, return_sequences=True))
    model.add(LSTM(hidden_units, kernel_initializer='he_normal', activation='tanh', dropout=dropout))
    model.add(Dense(2, activation='softmax'))
    return model


# A model using just Gated Recurrent Units
def build_gru_model(max_length, vocab_size, embedding_vector_dim=256, hidden_units=256, dropout=0.3):
    print("GRU Model")

    model = Sequential()

    # Add an embedding layer to obtain a dense vector representation of the data
    # Using Glorot normal initializer (also called Xavier normal initializer)
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_vector_dim, input_length=max_length,
                        embeddings_initializer='glorot_normal'))
    # GRU Layers
    model.add(GRU(hidden_units, kernel_initializer='he_normal', activation='tanh', dropout=dropout, return_sequences=True))
    model.add(GRU(hidden_units, kernel_initializer='he_normal', activation='tanh', dropout=dropout))
    model.add(Dense(2, activation='softmax'))
    return model


def build_bidirectional_model(max_length, vocab_size, embedding_vector_dim=256, hidden_units=256, dropout=0.3):
    print("Bidirectional LSTM Model")

    model = Sequential()

    # Add an embedding layer to obtain a dense vector representation of the data
    # Using Glorot normal initializer (also called Xavier normal initializer)
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_vector_dim, input_length=max_length,
                        embeddings_initializer='glorot_normal'))  # mask_zero=True))
    model.add(LSTM(hidden_units, kernel_initializer='he_normal', activation='tanh', dropout=dropout, return_sequences=True))
    # Bidirectional wrapper for LSTM
    model.add(Bidirectional(LSTM(hidden_units, kernel_initializer='he_normal', activation='tanh', dropout=dropout)))
    model.add(Dense(2, activation='softmax'))
    return model


# This is the precise architecture as Ghosh has proposed in his paper "Fracking sarcasm"
def build_conv_and_lstm_model(max_length, vocab_size, embedding_vector_dim=256, hidden_units=256, dropout=0.3):
    print("CONV + LSTM Model")

    model = Sequential()

    # Add an embedding layer to obtain a dense vector representation of the data
    # Using Glorot normal initializer (also called Xavier normal initializer)
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_vector_dim, input_length=max_length,
                        embeddings_initializer='glorot_normal'))  # mask_zero=True))
    model.add(Conv1D(hidden_units, 3, kernel_initializer='he_normal', padding='valid', activation='relu',
                            input_shape=(1, max_length)))
    model.add(Conv1D(hidden_units, 3, kernel_initializer='he_normal', padding='valid', activation='relu',
                            input_shape=(1, max_length - 2)))
    model.add(LSTM(hidden_units, kernel_initializer='he_normal', activation='tanh', dropout=dropout, return_sequences=True))
    model.add(LSTM(hidden_units, kernel_initializer='he_normal', activation='tanh', dropout=dropout))
    model.add(Dense(hidden_units, kernel_initializer='he_normal', activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    return model


def predict(model, x_test, y_test):
    y = []
    y_pred = []
    prediction_probability = model.predict_proba(x_test, batch_size=1, verbose=0)
    print("Predicted probability length: ", len(prediction_probability))
    for i, (label) in enumerate(prediction_probability):
        predicted = np.argmax(prediction_probability[i])
        y.append(int(y_test[i]))
        y_pred.append(predicted)
    utils.print_statistics(y, y_pred)


def print_settings(max_tweet_length, vocab_size, embedding_vector_dim, hidden_units,
                   epochs, batch_size, dropout):
    print("==================================================================\n")
    print("Model Settings:\n")
    print("==================================================================\n")
    print("Max tweet length = ", max_tweet_length)
    print("Vocab size = ", vocab_size)
    print("Embedding vector dimension = ", embedding_vector_dim)
    print("Hidden units ", hidden_units)
    print("Epochs ", epochs)
    print("Batch size ", batch_size)
    print("Dropout ", dropout)
    print("==================================================================\n")


class writer:
    def __init__(self, *writers):
        self.writers = writers

    def write(self, text):
        for w in self.writers:
            w.write(text)

    def flush(self):
        pass


if __name__ == "__main__":
    path = os.getcwd()[:os.getcwd().rfind('/')]
    saved = sys.stdout
    fout = open(path + '/stats/dnn_models_analysis_second_run.txt', 'wt')
    sys.stdout = writer(sys.stdout, fout)
    print("Current date and time: %s\n" % str(datetime.datetime.now()))

    # sys.stdout = open(path + '/stats/dnn_models_analysis.txt', 'wt')
    train_file = "train_sample.txt"
    dev_file = "dev.txt"
    test_file = "test_sample.txt"
    word_list = "word_list.txt"

    # For each tweet, get the filtered data, the sequences of word indices and the labels
    train_tweets, train_indices, train_labels, \
    dev_tweets, dev_indices, dev_labels, \
    test_tweets, test_indices, test_labels, vocab_size = \
        data_proc.get_clean_dl_data(train_file, dev_file, test_file, word_list)

    # Get some idea about the max length of the train tweets
    max_tweet_length = utils.get_max_len_info(train_tweets)

    # Calculate the ratio of classes to solve the imbalance
    ratio = utils.get_classes_ratio(train_labels)

    plot_training_graph = True

    # Build and analyse the models
    dnn_models = ['standard', 'conv', 'lstm', 'gru', 'bidirectional', 'conv+lstm']
    dnn_models = ['conv']
    # Pad sequences with 0s
    x_train = sequence.pad_sequences(train_indices, maxlen=max_tweet_length, padding='pre', truncating='pre', value=0.)
    x_dev = sequence.pad_sequences(dev_indices, maxlen=max_tweet_length, padding='pre', truncating='pre', value=0.)
    x_test = sequence.pad_sequences(test_indices, maxlen=max_tweet_length, padding='pre', truncating='pre', value=0.)

    # Transform the output into categorical data
    y_train, y_dev = [np_utils.to_categorical(y) for y in (train_labels, dev_labels)]

    # The settings for the upcoming models
    epochs = 5
    batch_size = 8
    embedding_vector_dim = 256
    hidden_units = 100
    dropout = 0.3
    print_settings(max_tweet_length, vocab_size, embedding_vector_dim, hidden_units, epochs, batch_size, dropout)

    # Build the model
    for dnn_model in dnn_models:
        start = time.time()

        # Build the deep neural network architecture
        if dnn_model == 'standard':
            model = build_standard_dnn_model(x_train.shape[1], hidden_units)
        elif dnn_model == 'conv':
            model = build_conv_model(max_tweet_length, vocab_size, embedding_vector_dim, hidden_units, dropout)
        elif dnn_model == 'lstm':
            model = build_lstm_model(max_tweet_length, vocab_size, embedding_vector_dim, hidden_units, dropout)
        elif dnn_model == 'gru':
            model = build_gru_model(max_tweet_length, vocab_size, embedding_vector_dim, hidden_units, dropout)
        elif dnn_model == 'bidirectional':
            model = build_bidirectional_model(max_tweet_length, vocab_size, embedding_vector_dim, hidden_units, dropout)
        elif dnn_model == 'conv+lstm':
            model = build_conv_and_lstm_model(max_tweet_length, vocab_size, embedding_vector_dim, hidden_units)

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001),
                      metrics=['accuracy', utils.f1_score])

        # Print the model summary
        print(model.summary())

        # Save an image of the current architecture
        plot_model(model, to_file=path + '/models/dnn_models/' + dnn_model + '_model_summary.png',
                   show_shapes=True, show_layer_names=True)

        # Save the json representation of the model
        open(path + '/models/dnn_models/model_json/' + dnn_model + '_model.json', 'w').write(model.to_json())

        # Prepare the callbacks
        save_best = ModelCheckpoint(monitor='val_loss', filepath=path + '/models/dnn_models/best/' + dnn_model +
                                    '_model.json.hdf5', save_best_only=True)
        save_all = ModelCheckpoint(monitor='val_loss', filepath=path + '/models/dnn_models/checkpoints/' + dnn_model +
                                   '_model_weights_epoch_{epoch:02d}_acc_{val_acc:.3f}_f_{val_f1_score:.3f}.hdf5',
                                   save_best_only=False)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')

        # Fit the model on the training data
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, class_weight=ratio,
                            validation_data=(x_dev, y_dev), callbacks=[save_best, save_all, early_stopping], verbose=1)

        # If want to plot model
        if plot_training_graph:
            utils.plot_training_statistics(history, path + "/plots/dnn_models/" + dnn_model + "_model.png")

        # Load the best model
        model = utils.load_model(json_name=path + '/models/dnn_models/model_json/' + dnn_model + '_model.json',
                                 h5_weights_name=path + '/models/dnn_models/best/' + dnn_model + '_model.json.hdf5')

        # Make prediction and evaluation
        predict(model, x_test, test_labels)

        end = time.time()
        print("==================================================================\n")
        print("%s model analysis completion time: %.3f s = %.3f min"
              % (dnn_model, (end - start), (end - start) / 60.0))
        print("==================================================================\n")
    sys.stdout = saved
    fout.close()