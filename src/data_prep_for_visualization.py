from __future__ import print_function
from glob import glob
from natsort import natsorted
from keras.models import load_model
import utils, keras, os, shutil
import numpy as np
import data_processing as data_proc
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, LSTM
from keras.layers import Dense, Flatten
from keras.models import Sequential
from visualize_hidden_units import get_activations, visualize_activations

num_classes = 2


# Prepare data for visualization -- here grammatical data
# i.e all hashtags split, all emojis kept, everything in lower-case (see data_processing)
def get_data():
    path = os.getcwd()[:os.getcwd().rfind('/')]
    train_filename = "train.txt"
    test_filename = "test.txt"
    word_list = "word_list.txt"
    dict_list = "word_list_freq.txt"

    # Load the train and test sets
    print("Loading data...")
    train_tweets, train_labels = utils.load_data_panda(path + "/res/datasets/" + train_filename)
    test_tweets, test_labels = utils.load_data_panda(path + "/res/datasets/" + test_filename)

    # Initial clean of data
    data_proc.initial_clean(train_tweets, path + "/res/datasets/grammatical_" + train_filename, word_list,
                            word_file_is_dict=True, split_hashtag_method=data_proc.split_hashtags2)
    data_proc.initial_clean(test_tweets, path + "/res/datasets/grammatical_" + test_filename, word_list,
                            word_file_is_dict=True, split_hashtag_method=data_proc.split_hashtags2)

    # Get the tags corresponding to the test and train files
    train_tokens, train_pos = \
        data_proc.get_tags_for_each_tweet(path + "/res/cmu_tweet_tagger/grammatical_tweets_" + train_filename,
                                          path + "/res/tokens/grammatical_tokens_" + train_filename,
                                          path + "/res/pos/grammatical_pos_" + train_filename)
    test_tokens, test_pos = \
        data_proc.get_tags_for_each_tweet(path + "/res/cmu_tweet_tagger/grammatical_tweets_" + test_filename,
                                          path + "/res/tokens/grammatical_tokens_" + test_filename,
                                          path + "/res/pos/grammatical_pos_" + test_filename)

    # Clean the data and brind it to the most *grammatical* form possible
    gramm_train_tokens = data_proc.grammatical_clean \
        (train_tokens, train_pos, path + "/res/" + dict_list,
         path + "/res/grammatical/clean_for_grammatical_tweets_no_emoji_replacement_" + train_filename,
         replace_emojis=False)
    gramm_test_tokens = data_proc.grammatical_clean \
        (test_tokens, test_pos, path + "/res/" + dict_list,
         path + "/res/grammatical/clean_for_grammatical_tweets_no_emoji_replacement_" + test_filename,
         replace_emojis=False)

    # Make all words lower-case
    gramm_train_tokens = [t.lower() for t in gramm_train_tokens]
    gramm_test_tokens = [t.lower() for t in gramm_test_tokens]

    # Get the max length of the train tweets
    max_tweet_length = utils.get_max_len_info(train_tweets)

    # Convert all tweets into sequences of word indices
    tokenizer, train_indices, test_indices = utils.encode_text_as_word_indexes(gramm_train_tokens, gramm_test_tokens,
                                                                               lower=True)
    vocab_size = len(tokenizer.word_counts) + 1
    word_to_index = tokenizer.word_index
    print('There are %s unique tokens.' % len(word_to_index))

    # Pad sequences with 0s
    x_train = pad_sequences(train_indices, maxlen=max_tweet_length, padding='post', truncating='post', value=0.)
    x_test = pad_sequences(test_indices, maxlen=max_tweet_length, padding='post', truncating='post', value=0.)

    # Transform the output into categorical data
    y_train = to_categorical(np.asarray(train_labels))
    y_test = to_categorical(np.asarray(test_labels))
    return x_train, y_train, x_test, y_test, vocab_size, tokenizer, max_tweet_length


if __name__ == '__main__':
    checkpoints = glob('checkpoints/*.h5')
    if len(checkpoints) > 0:
        # Get the previously trained model
        checkpoints = natsorted(checkpoints)
        assert len(checkpoints) != 0, 'No checkpoints found.'
        checkpoint_file = checkpoints[-1]
        print('Loading [{}]'.format(checkpoint_file))
        model = load_model(checkpoint_file)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print(model.summary())

        # Load the data
        x_train, y_train, x_test, y_test, vocab_size, tokenizer, max_tweet_length = get_data()

        # Get the word to index and the index to word mappings
        word_index = tokenizer.word_index
        index_to_word = {index: word for word, index in word_index.items()}

        # Evaluate the previously trained model on test data
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1, batch_size=8)
        print('Loss ', test_loss, ' accuracy ', test_acc)

        # Visualize the activations
        vis_input = x_test[3473:3474]   # Select a testing example to visualize from the test dataset
        activations, names = get_activations(model, vis_input, layer_name=None)
        visualize_activations(activations, names, vis_input, index_to_word)

    else:
        x_train, y_train, x_test, y_test, vocab_size, tokenizer, max_tweet_length = get_data()
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_tweet_length,
                            embeddings_initializer='glorot_normal', name='embedding_layer'))
        model.add(LSTM(output_dim=32, name="recurrent_layer", activation="tanh", return_sequences=True))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', name='dense_layer'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        model.summary()
        try:
            shutil.rmtree('checkpoints')
        except:
            pass
        os.mkdir('checkpoints')
        checkpoint = ModelCheckpoint(monitor='val_acc',
                                     filepath='checkpoints/model_{epoch:02d}_{val_acc:.3f}.h5',
                                     save_best_only=True)
        model.fit(x_train, y_train,
                  batch_size=8,
                  epochs=4,
                  verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks=[checkpoint])
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
