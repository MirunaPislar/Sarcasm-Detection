import os, sys, time
import numpy as np
import data_processing as data_proc
import utils
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, LSTM, GRU, Bidirectional, Input
from keras.layers import Activation, Permute, RepeatVector, TimeDistributed, Merge, Lambda
from keras.utils import to_categorical
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from numpy.random import seed
seed(1337603)


# A standard DNN used as a baseline
def standard_dnn_model(embeddings, hidden_units=256, dropout=0.3):
    X = Dense(hidden_units, kernel_initializer='he_normal', activation='relu')(embeddings)
    X = Flatten()(X)
    return X


# A model using just convolutional neural networks
def cnn_model(embeddings, hidden_units=256, dropout=0.3):
    X = Conv1D(filters=hidden_units, kernel_size=3, kernel_initializer='he_normal', padding='valid',
               activation='relu')(embeddings)
    X = Conv1D(filters=hidden_units, kernel_size=3, kernel_initializer='he_normal', padding='valid',
               activation='relu')(X)
    X = GlobalMaxPooling1D()(X)
    # X = MaxPooling1D(pool_size=3)(X)
    # X = Flatten()(X)
    return X


# A model using Long Short Term Memory (LSTM) Units
def lstm_model(embeddings, hidden_units=256, dropout=0.3):
    X = LSTM(hidden_units, kernel_initializer='he_normal', activation='tanh', return_sequences=True)(embeddings)
    X = Dropout(dropout)(X)
    X = LSTM(hidden_units, kernel_initializer='he_normal', activation='tanh', return_sequences=False)(X)
    X = Dropout(dropout)(X)
    return X


# A model using just Gated Recurrent Units (GRU)
def gru_model(embeddings, hidden_units=256, dropout=0.3):
    X = GRU(hidden_units, kernel_initializer='he_normal', activation='tanh', return_sequences=True)(embeddings)
    X = Dropout(dropout)(X)
    X = GRU(hidden_units, kernel_initializer='he_normal', activation='tanh', return_sequences=False)(X)
    X = Dropout(dropout)(X)
    return X


# A model using a bidirectional LSTM deep network
def bidirectional_lstm_model(embeddings, hidden_units=256, dropout=0.3):
    X = LSTM(hidden_units, kernel_initializer='he_normal', activation='tanh', return_sequences=True)(embeddings)
    X = Dropout(dropout)(X)
    X = Bidirectional(LSTM(hidden_units, kernel_initializer='he_normal', activation='tanh', return_sequences=False))(X)
    X = Dropout(dropout)(X)
    return X


# This is the precise architecture as Ghosh has proposed in his paper "Fracking sarcasm"
def cnn_lstm_model(embeddings, hidden_units=256, dropout=0.3):
    X = Conv1D(hidden_units, 3, kernel_initializer='he_normal', padding='valid', activation='relu')(embeddings)
    X = Conv1D(hidden_units, 3, kernel_initializer='he_normal', padding='valid', activation='relu')(X)
    X = LSTM(hidden_units, kernel_initializer='he_normal', activation='tanh', dropout=dropout, return_sequences=True)(X)
    X = LSTM(hidden_units, kernel_initializer='he_normal', activation='tanh', dropout=dropout)(X)
    X = Dense(hidden_units, kernel_initializer='he_normal', activation='sigmoid')(X)
    return X


# Create a Keras embedding layer of pre-trained global vectors
def pretrained_embedding_layer(word2vec_map, word_to_index, trainable=False):
    embedding_dim = word2vec_map["cucumber"].shape[0]
    embedding_matrix = utils.get_embeding_matrix(word2vec_map, word_to_index, embedding_dim)
    # Add an embedding layer but prevent the weights from being updated during training.
    embedding_layer = Embedding(vocab_size, embedding_dim, trainable=trainable)
    # Build the embedding layer, it is required before setting the weights of the embedding layer.
    embedding_layer.build((None,))
    # Set the weights of the embedding layer to the embedding matrix.
    embedding_layer.set_weights([embedding_matrix])
    return embedding_layer


def build_model(max_tweet_length, vocab_size, embedding_dim, hidden_units, dropout,
                dnn_architecture, use_glove_embeddings=False):
    # Define sentence_indices as the input of the graph (of type int, since it contains indices)
    tweet_indices = Input((max_tweet_length,), dtype='int32')
    if use_glove_embeddings:
        word2vec_map = utils.load_glove_vectors(glove_filename='glove.6B.100d.txt')
        # Create the embedding layer pretrained with GloVe Vectors
        embedding_layer = pretrained_embedding_layer(word2vec_map, word_to_index)
        # Propagate sentence_indices through your embedding layer, you get back the embeddings
        embeddings = embedding_layer(tweet_indices)
    else:
        embedding_layer = Embedding(vocab_size, embedding_dim, input_length=max_tweet_length)
        # Build the embedding layer, it is required before setting the weights of the embedding layer.
        embedding_layer.build((None,))
        embeddings = embedding_layer(tweet_indices)
    X = dnn_architecture(embeddings, hidden_units, dropout)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 2-dimensional vectors.
    X = Dense(2)(X)
    # Add a softmax activation
    X = Activation('softmax')(X)
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=tweet_indices, outputs=X)
    return model


def predict(model, x_test, y_test):
    y = []
    y_pred = []
    prediction_probability = model.predict(x_test)
    print("Predicted probability length: ", len(prediction_probability))
    for i, (label) in enumerate(prediction_probability):
        predicted = np.argmax(prediction_probability[i])
        y.append(int(y_test[i]))
        y_pred.append(predicted)
    utils.print_statistics(y, y_pred)


# Dictionary to look up the names and architectures for different models
def dnn_options(name):
    return {
        'standard': standard_dnn_model,
        'cnn': cnn_model,
        'lstm': lstm_model,
        'gru': gru_model,
        'bidirectional': bidirectional_lstm_model,
        'cnn+lstm': cnn_lstm_model,
        'attention': attention_model2
    }[name]


if __name__ == "__main__":
    path = os.getcwd()[:os.getcwd().rfind('/')]
    to_write_filename = path + '/stats/dnn_models_analysis_19feb.txt'
    utils.initialize_writer(to_write_filename)

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
    gramm_train_tokens = data_proc.grammatical_clean\
        (train_tokens, train_pos, path + "/res/" + dict_list,
         path + "/res/grammatical/clean_for_grammatical_tweets_no_emoji_replacement_" + train_filename,
         replace_emojis=False)
    gramm_test_tokens = data_proc.grammatical_clean\
        (test_tokens, test_pos, path + "/res/" + dict_list,
         path + "/res/grammatical/clean_for_grammatical_tweets_no_emoji_replacement_" + test_filename,
         replace_emojis=False)

    # Make all words lower-case
    gramm_train_tokens = [t.lower() for t in gramm_train_tokens]
    gramm_test_tokens = [t.lower() for t in gramm_test_tokens]

    # Get the max length of the train tweets
    max_tweet_length = utils.get_max_len_info(train_tweets)

    # Convert all tweets into sequences of word indices
    tokenizer, train_indices, test_indices = utils.encode_text_as_word_indexes(gramm_train_tokens, gramm_test_tokens, lower=True)
    vocab_size = len(tokenizer.word_counts) + 1
    word_to_index = tokenizer.word_index
    print('There are %s unique tokens.' % len(word_to_index))

    # Pad sequences with 0s
    x_train = pad_sequences(train_indices, maxlen=max_tweet_length, padding='pre', truncating='pre', value=0.)
    x_test = pad_sequences(test_indices, maxlen=max_tweet_length, padding='pre', truncating='pre', value=0.)

    # Transform the output into categorical data
    y_train = to_categorical(np.asarray(train_labels))
    y_test = to_categorical(np.asarray(test_labels))

    portion_for_dev = 20
    x_dev = x_train[-portion_for_dev:]
    y_dev = y_train[-portion_for_dev:]
    x_train = x_train[:-portion_for_dev]
    y_train = y_train[:-portion_for_dev]

    print("Positive examples in train: ", sum(y_train[:, 1]))
    print("Negative examples in train: ", sum(y_train[:, 0]))
    print("Positive examples in dev: ", sum(y_dev[:, 1]))
    print("Negative examples in train: ", sum(y_dev[:, 0]))

    print("Shape of the x train set ", x_train.shape)
    print("Shape of the x dev set ", x_dev.shape)
    print("Shape of the x test set ", x_test.shape)
    print("Shape of the y train set ", y_train.shape)
    print("Shape of the y dev set ", y_dev.shape)
    print("Shape of the y test set ", y_test.shape)

    # Calculate the ratio of classes to solve the imbalance
    ratio = utils.get_classes_ratio(train_labels[:-portion_for_dev])

    plot_training_graph = False

    # Build and analyse the models
    dnn_models = ['standard', 'cnn', 'lstm', 'gru', 'bidirectional', 'cnn+lstm']

    # The settings for the upcoming models
    epochs = 100
    batch_size = 8
    embedding_vector_dim = 256
    hidden_units = 256
    dropout = 0.3
    utils.print_settings(max_tweet_length, vocab_size, embedding_vector_dim, hidden_units, epochs, batch_size, dropout)

    # Build the model
    for dnn_model in dnn_models:
        start = time.time()
        # Build the deep neural network architecture
        model = build_model(max_tweet_length, vocab_size, embedding_vector_dim, hidden_units, dropout,
                            dnn_architecture=dnn_options(dnn_model), use_glove_embeddings=True)
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
                                    '_model.json.hdf5', save_best_only=True, mode='min')
        save_all = ModelCheckpoint(monitor='val_loss', filepath=path + '/models/dnn_models/checkpoints/' + dnn_model +
                                   '_model_weights_epoch_{epoch:02d}_acc_{val_acc:.3f}_f_{val_f1_score:.3f}.hdf5',
                                    save_best_only=False, mode='min')
        reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')

        # Fit the model on the training data
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, class_weight=ratio,
                            validation_data=(x_dev, y_dev), callbacks=[save_best, early_stopping, reduceLR],  verbose=1)

        # If want to plot model
        if plot_training_graph:
            utils.plot_training_statistics(history, path + "/plots/dnn_models/" + dnn_model)

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
