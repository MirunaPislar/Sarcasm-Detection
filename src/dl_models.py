import os, sys, time, utils
import numpy as np
import data_processing as data_proc
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, LSTM, GRU, Bidirectional, Input, Multiply
from keras.engine.topology import Layer
from keras.layers import Activation, Permute, RepeatVector, Lambda
from keras.utils import to_categorical
import keras.backend as K
from keras import initializers, activations
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from numpy.random import seed
seed(1337603)


# A standard DNN used as a baseline
def standard_dnn_model(**kwargs):
    X = Dense(kwargs['hidden_units'], kernel_initializer='he_normal', activation='relu')(kwargs['embeddings'])
    X = Flatten()(X)
    return X


# A model using just convolutional neural networks
def cnn_model(**kwargs):
    X = Conv1D(filters=kwargs['hidden_units'], kernel_size=3, kernel_initializer='he_normal', padding='valid',
               activation='relu')(kwargs['embeddings'])
    X = Conv1D(filters=kwargs['hidden_units'], kernel_size=3, kernel_initializer='he_normal', padding='valid',
               activation='relu')(X)
    X = GlobalMaxPooling1D()(X)
    # X = MaxPooling1D(pool_size=3)(X)
    # X = Flatten()(X)
    return X


# A model using Long Short Term Memory (LSTM) Units
def lstm_model(**kwargs):
    X = LSTM(kwargs['hidden_units'], kernel_initializer='he_normal', activation='tanh',
             dropout=kwargs['dropout'], return_sequences=True)(kwargs['embeddings'])
    X = LSTM(kwargs['hidden_units'], kernel_initializer='he_normal', activation='tanh',
             dropout=kwargs['dropout'], return_sequences=True)(X)
    X = Flatten()(X)
    return X


# A model using just Gated Recurrent Units (GRU)
def gru_model(**kwargs):
    X = GRU(kwargs['hidden_units'], kernel_initializer='he_normal', activation='tanh',
            dropout=kwargs['dropout'], return_sequences=True)(kwargs['embeddings'])
    X = GRU(kwargs['hidden_units'], kernel_initializer='he_normal', activation='tanh',
            dropout=kwargs['dropout'], return_sequences=False)(X)
    return X


# A model using a bidirectional LSTM deep network
def bidirectional_lstm_model(**kwargs):
    X = LSTM(kwargs['hidden_units'], kernel_initializer='he_normal', activation='tanh',
             dropout=kwargs['dropout'], return_sequences=True)(kwargs['embeddings'])
    X = Bidirectional(LSTM(kwargs['hidden_units'], kernel_initializer='he_normal', activation='sigmoid',
                           dropout=kwargs['dropout'], return_sequences=False))(X)
    return X


# This is the precise architecture as Ghosh has proposed in his paper "Fracking sarcasm"
def cnn_lstm_model(**kwargs):
    X = Conv1D(kwargs['hidden_units'], 3, kernel_initializer='he_normal', padding='valid', activation='relu')(kwargs['embeddings'])
    X = Conv1D(kwargs['hidden_units'], 3, kernel_initializer='he_normal', padding='valid', activation='relu')(X)
    X = LSTM(kwargs['hidden_units'], kernel_initializer='he_normal', activation='tanh',
             dropout=kwargs['dropout'], return_sequences=True)(X)
    X = LSTM(kwargs['hidden_units'], kernel_initializer='he_normal', activation='tanh',
             dropout=kwargs['dropout'])(X)
    X = Dense(kwargs['hidden_units'], kernel_initializer='he_normal', activation='sigmoid')(X)
    return X


# This is a pretty simple architecture for an LSTM network with a 'stateless' attention layer on top
def stateless_attention_model(**kwargs):
    X = LSTM(kwargs['hidden_units'], kernel_initializer='he_normal', activation='tanh',
             dropout=kwargs['dropout'], return_sequences=True)(kwargs['embeddings'])
    attention_layer = Permute((2, 1))(X)
    attention_layer = Dense(kwargs['max_tweet_length'], activation='softmax')(attention_layer)
    attention_layer = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(attention_layer)
    attention_layer = RepeatVector(int(X.shape[2]))(attention_layer)
    attention_probabilities = Permute((2, 1), name='attention_probs')(attention_layer)
    attention_layer = Multiply()([X, attention_probabilities])
    attention_layer = Flatten()(attention_layer)
    return attention_layer


class MyAttentionLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('glorot_uniform')
        super(MyAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Make sure it receives a 3D tensor with shape (batch_size, timesteps, input_dim)
        assert len(input_shape) == 3
        # Create a trainable weight variable for this layer.
        self.a = self.add_weight((input_shape[-1],), initializer=self.init, name='lstm_weight')
        self.trainable_weights = [self.a]
        super(MyAttentionLayer, self).build(input_shape)

    def call(self, x):
        # Insert a dimension of 1 at the last index to the tensor
        expanded_a = K.expand_dims(self.a)
        eij = K.tanh(K.squeeze(K.dot(x, expanded_a), axis=-1))
        ai = K.exp(eij)
        attention_weights = ai / K.cast(K.sum(ai, axis=1, keepdims=True), K.floatx())
        # Insert a dimension of 1 at the last index to the tensor
        attention_weights = K.expand_dims(attention_weights)
        context = x * attention_weights
        return K.sum(context, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def attention_model(**kwargs):
    lstm_out = LSTM(kwargs['hidden_units'], kernel_initializer='he_normal', activation='tanh',
                    dropout=kwargs['dropout'], return_sequences=True)(kwargs['embeddings'])
    attention = MyAttentionLayer()(lstm_out)
    return attention


def pretrained_embedding_layer(word2vec_map, word_to_index, embedding_dim, vocab_size, trainable=False):
    embedding_matrix = utils.get_embedding_matrix(word2vec_map, word_to_index, embedding_dim)
    embedding_layer = Embedding(vocab_size, embedding_dim, trainable=trainable)
    embedding_layer.build((None,))
    embedding_layer.set_weights([embedding_matrix])
    return embedding_layer


def build_embedding_layer(word2index, emb_type='glove', embedding_dim=300, max_len=40, trainable=True):
    vocab_size = len(word2index) + 1
    if 'glove' in emb_type:
        word2vec_map = utils.load_vectors(filename='glove.6B.%dd.txt' % embedding_dim)
        emb_layer = pretrained_embedding_layer(word2vec_map, word2index, embedding_dim, vocab_size, trainable=trainable)
    elif 'emoji' in emb_type:
        emoji2vec_map = utils.load_vectors(filename='emoji_embeddings_%dd.txt' % embedding_dim)
        emb_layer = pretrained_embedding_layer(emoji2vec_map, word2index, embedding_dim, vocab_size, trainable=trainable)
    elif 'random' in emb_type:
        words = word2index.keys()
        random2vec_map = utils.build_random_word2vec(words, embedding_dim=embedding_dim, variance=1)
        emb_layer = pretrained_embedding_layer(random2vec_map, word2index, embedding_dim, vocab_size, trainable=trainable)
    else:
        emb_layer = Embedding(vocab_size, embedding_dim, input_length=max_len, trainable=trainable)
        emb_layer.build((None,))
    return emb_layer


def build_model(max_len, embedding_layer, hidden_units, dropout, dnn_architecture):
    tweet_indices = Input((max_len,), dtype='int32')
    embeddings = embedding_layer(tweet_indices)
    X = dnn_architecture(max_tweet_length=max_len, embeddings=embeddings, hidden_units=hidden_units, dropout=dropout)
    X = Dense(hidden_units, kernel_initializer='he_normal', activation='relu')(X)
    X = Dense(2)(X)
    X = Activation('softmax')(X)
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
        'Standard': standard_dnn_model,
        'CNN': cnn_model,
        'LSTM': lstm_model,
        'GRU': gru_model,
        'Bidirectional LSTM': bidirectional_lstm_model,
        'CNN + LSTM': cnn_lstm_model,
        'Stateless Attention': stateless_attention_model,
        'Attention': attention_model,
    }[name]


def run_dl_analysis(train_tweets, test_tweets, y_train, y_test, path, shuffle=True,
                    max_tweet_length=40, emb_type='glove', trainable=True, plot=True,
                    dnn_models=None, epochs=50, batch_size=32, embedding_dim=300, hidden_units=256, dropout=0.5):
    if shuffle:
        train_tweets = utils.shuffle_words(train_tweets)
        test_tweets = utils.shuffle_words(test_tweets)

    # Convert all tweets into sequences of word indices
    tokenizer, train_indices, test_indices = utils.encode_text_as_word_indexes(train_tweets, test_tweets, lower=True)
    word_to_index = tokenizer.word_index
    print('There are %s unique tokens.' % len(word_to_index))

    # Pad sequences with 0s
    x_train = pad_sequences(train_indices, maxlen=max_tweet_length, padding='post', truncating='post', value=0.)
    x_test = pad_sequences(test_indices, maxlen=max_tweet_length, padding='post', truncating='post', value=0.)

    print("Shape of the x train set ", x_train.shape)
    print("Shape of the x test set ", x_test.shape)

    ratio = utils.get_classes_ratio(train_labels)

    # Define the embedding layer (which will be the same for all the models)
    embedding_layer = build_embedding_layer(word_to_index, emb_type, embedding_dim, max_tweet_length, trainable)

    # Build the model
    for dnn_model in dnn_models:
        start = time.time()

        # Build the deep neural network architecture
        utils.print_model_title(dnn_model)
        model = build_model(max_tweet_length, embedding_layer, hidden_units, dropout, dnn_architecture=dnn_options(dnn_model))

        # Compile the model
        my_optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.99, decay=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=my_optimizer, metrics=['accuracy', utils.f1_score])

        # Print the model summary
        print(model.summary())

        if plot:            # to save an image of the current architecture
            plot_model(model, to_file=path + '/models/dnn_models/' + dnn_model.lower() + '_model_summary.png',
                       show_shapes=True, show_layer_names=True)

        # Save the json representation of the model
        open(path + '/models/dnn_models/model_json/' + dnn_model.lower() + '_model.json', 'w').write(model.to_json())

        # Prepare the callbacks
        save_best = ModelCheckpoint(monitor='val_categorical_accuracy', save_best_only=True, mode='auto',
                                    filepath=path + '/models/dnn_models/best/' + dnn_model.lower() + '_model.json.hdf5')
        reduceLR = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=1)

        # Fit the model on the training data
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, class_weight=ratio,
                            callbacks=[save_best, reduceLR, early_stopping], validation_split=0.1, verbose=1)

        if plot:
            utils.plot_training_statistics(history, "/plots/dnn_models/" + dnn_model)

        # Load the best model
        model = utils.load_model(json_name=path + '/models/dnn_models/model_json/' + dnn_model.lower() + '_model.json',
                                 h5_weights_name=path + '/models/dnn_models/best/' + dnn_model.lower() + '_model.json.hdf5')

        # Make prediction and evaluation
        predict(model, x_test, y_test)
        end = time.time()
        print("==================================================================\n")
        print("%s model analysis completion time: %.3f s = %.3f min"
              % (dnn_model, (end - start), (end - start) / 60.0))
        print("==================================================================\n")


if __name__ == "__main__":
    path = os.getcwd()[:os.getcwd().rfind('/')]
    to_write_filename = path + '/stats/dnn_models_analysis.txt'
    utils.initialize_writer(to_write_filename)

    train_filename = "train_sample.txt"
    test_filename = "test_sample.txt"

    # Load the train and test sets
    print("Loading data...")
    train_data = utils.load_file(path + "/res/tokens/tokens_clean_original_" + train_filename).split("\n")
    test_data = utils.load_file(path + "/res/tokens/tokens_clean_original_" + test_filename).split("\n")

    # Load the deepmoji predictions for each tweet
    train_deepmoji = utils.get_deepmojis("data_frame_" + train_filename[:-4] + ".csv", threshold=0.05)
    train_deepmoji = [' '.join(e) for e in train_deepmoji]

    test_deepmoji = utils.get_deepmojis("data_frame_" + test_filename[:-4] + ".csv", threshold=0.05)
    test_deepmoji = [' '.join(e) for e in test_deepmoji]

    # Load the labels
    train_labels = [int(l) for l in utils.load_file(path + "/res/data/labels_" + train_filename).split("\n")]
    test_labels = [int(l) for l in utils.load_file(path + "/res/data/labels_" + test_filename).split("\n")]

    # Transform the output into categorical data
    y_train = to_categorical(np.asarray(train_labels))
    y_test = test_labels

    # Make and print the settings for the DL model
    max_len = utils.get_max_len_info(train_data)
    emb_type = 'emoji'
    trainable = True
    plot = True
    shuffle = True
    epochs = 5
    batch_size = 16
    embedding_dim = 50
    hidden_units = 256
    dropout = 0.3
    utils.print_settings(max_len, embedding_dim, hidden_units, epochs, batch_size, dropout, emb_type, trainable)

    if 'emoji' in emb_type:
        train_data = train_deepmoji
        test_data = test_deepmoji

    # List of the models to be analysed
    models = ['Standard', 'CNN', 'LSTM', 'GRU', 'Bidirectional LSTM', 'CNN + LSTM', 'Stateless Attention', 'Attention']
    models = ['Standard']

    # Run model
    run_dl_analysis(train_data, test_data, y_train, y_test, path, shuffle, max_len, emb_type,
                    trainable, plot, models, epochs, batch_size, embedding_dim, hidden_units, dropout)
