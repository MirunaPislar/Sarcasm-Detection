from __future__ import print_function
from glob import glob
from natsort import natsorted
from keras.models import load_model
import utils, keras, os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, LSTM
from keras.layers import Dense, Flatten
from keras.models import Sequential
from visualize_hidden_units import get_activations, visualize_activations

# Define some parameters
MODEL_PATH = os.getcwd()[:os.getcwd().rfind('/')] + '/models/dnn_models/vis_checkpoints/'
BATCH_SIZE = 256
EPOCHS = 10
EMBEDDING_DIM = 32
HIDDEN_UNITS = 32
DENSE_UNITS = 128
NO_OF_CLASSES = 2
SHUFFLE = False


# Prepare data for visualizations (attention and lstm)
def prepare_data(shuffle=False, labels_to_categorical=True):
    path = os.getcwd()[:os.getcwd().rfind("/")]
    to_write_filename = path + "/stats/data_prep_for_lstm_visualization.txt"
    utils.initialize_writer(to_write_filename)

    train_filename = "train.txt"
    test_filename = "test.txt"
    tokens_filename = "clean_original_"     # other types of tokens to experiment with in /res/tokens/
    data_path = path + "/res/tokens/tokens_"

    # Load the data
    train_data = utils.load_file(data_path + tokens_filename + train_filename)
    test_data = utils.load_file(data_path + tokens_filename + test_filename)

    if shuffle:
        train_data = utils.shuffle_words(train_data)
        test_data = utils.shuffle_words(test_data)
        print("DATA IS SHUFFLED")

    # Load the labels
    train_labels = [int(l) for l in utils.load_file(path + "/res/datasets/ghosh/labels_" + train_filename)]
    test_labels = [int(l) for l in utils.load_file(path + "/res/datasets/ghosh/labels_" + test_filename)]

    # Get the max length of the train tweets
    max_tweet_length = utils.get_max_len_info(train_data)

    # Convert all tweets into sequences of word indices
    tokenizer, train_indices, test_indices = utils.encode_text_as_word_indexes(train_data, test_data, lower=True)
    vocab_size = len(tokenizer.word_counts) + 1
    word_to_index = tokenizer.word_index
    print("There are %s unique tokens." % len(word_to_index))

    # Pad sequences with 0s (can do it post or pre - post works better here)
    x_train = pad_sequences(train_indices, maxlen=max_tweet_length, padding="post", truncating="post", value=0.)
    x_test = pad_sequences(test_indices, maxlen=max_tweet_length, padding="post", truncating="post", value=0.)

    # Transform the output into categorical data or just keep it as it is (in a numpy array)
    if labels_to_categorical:
        train_labels = to_categorical(np.asarray(train_labels))
        test_labels = to_categorical(np.asarray(test_labels))
    else:
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)
    return x_train, train_labels, x_test, test_labels, vocab_size, tokenizer, max_tweet_length


# Visualize the activations for one tweet
def one_tweet_visualization(model, x_test, index_to_word, tweet_number=3473, plot=True, verbose=False):
    vis_input = x_test[tweet_number: tweet_number + 1]
    activations, names = get_activations(model, vis_input, layer_name=None)
    visualize_activations(activations, names, vis_input, index_to_word, plot, verbose)


# Get the best previously trained model (saved in MODEL_PATH) otherwise train a new model
def train_lstm_for_visualization():
    checkpoints = glob(MODEL_PATH + "*.h5")
    if len(checkpoints) > 0:
        checkpoints = natsorted(checkpoints)
        assert len(checkpoints) != 0, "No checkpoints for visualization found."
        checkpoint_file = checkpoints[-1]
        print("Loading [{}]".format(checkpoint_file))
        model = load_model(checkpoint_file)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", utils.f1_score])
        print(model.summary())

        # Load the data
        x_train, y_train, x_test, y_test, vocab_size, tokenizer, max_tweet_length = prepare_data(SHUFFLE)

        # Get the word to index and the index to word mappings
        word_index = tokenizer.word_index
        index_to_word = {index: word for word, index in word_index.items()}

        # Evaluate the previously trained model on test data
        test_loss, test_acc, test_fscore = model.evaluate(x_test, y_test, verbose=1, batch_size=256)
        print("Loss: %.3f\nF-score: %.3f\n" % (test_loss, test_fscore))
        return model, index_to_word, x_test
    else:
        # Load the data
        x_train, y_train, x_test, y_test, vocab_size, tokenizer, max_tweet_length = prepare_data(SHUFFLE)

        # Get the word to index and the index to word mappings
        word_index = tokenizer.word_index
        index_to_word = {index: word for word, index in word_index.items()}

        # Build, evaluate and save the model
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_tweet_length,
                            embeddings_initializer="glorot_normal", name="embedding_layer"))
        model.add(LSTM(output_dim=HIDDEN_UNITS, name="recurrent_layer", activation="tanh", return_sequences=True))
        model.add(Flatten())
        model.add(Dense(DENSE_UNITS, activation="relu", name="dense_layer"))
        model.add(Dense(NO_OF_CLASSES, activation="softmax"))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                      metrics=["accuracy", utils.f1_score])
        model.summary()
        checkpoint = ModelCheckpoint(monitor="val_acc", filepath=MODEL_PATH + "model_{epoch:02d}_{val_acc:.3f}.h5",
                                     save_best_only=True, mode="max")
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                  validation_data=(x_test, y_test), callbacks=[checkpoint])
        score = model.evaluate(x_test, y_test)
        print("Loss: %.3f\nF-score: %.3f\n" % (score[0], score[1]))
        return model, index_to_word, x_test


if __name__ == "__main__":
    model, index_to_word, x_test = train_lstm_for_visualization()
    # Select a tweet number to plot visualizations for (will be saved as html and ust plotted (not saved) using matplot)
    one_tweet_visualization(model, x_test, index_to_word, tweet_number=3473, plot=False, verbose=True)
