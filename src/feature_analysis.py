import os, classifiers, utils
import data_processing as data_proc
import extract_statistical_features as extract_feature


def perform_feature_analysis(train_tokens, filtered_train_tokens, train_pos, filtered_train_pos, train_labels,
                             test_tokens, filtered_test_tokens, test_pos, filtered_test_pos, test_labels,
                             feature_set, features_names, activation_list, feature_function):

    if feature_set is not 'topic':
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
            utils.print_features(activation, features_names)

            # Select the active features, used in current analysis
            selected_features = [features_names[i] for i in range(0, len(features_names))
                                 if activation[i] is True]
            active_train_features = select_active_features(all_train_features, selected_features)
            active_test_features = select_active_features(all_test_features, selected_features)

            # Convert feature dictionary to a list of feature values (a dict vectorizer)
            train_features, test_features = utils.extract_features_from_dict(active_train_features, active_test_features)

            # Scale the features
            scaled_train_features = utils.feature_scaling(train_features)
            scaled_test_features = utils.feature_scaling(test_features)

            # Run the models
            utils.run_models(scaled_train_features, train_labels, scaled_test_features, test_labels)
    else:
        for activation in activation_list:
            # Print the current grid of selected features
            utils.print_features(activation, features_names)

            # Build the topic LDA model
            dictionary, corpus, lda_model = extract_feature.build_lda_model(train_tokens, train_pos,
                                           use_nouns=activation[2], use_verbs=activation[3], use_all=activation[4],
                                           num_of_topics=activation[0], passes=activation[1], verbose=False)
            # Get all the train and test features, each in their own dictionary of feature names:feature values
            train_features = []
            for index in range(len(train_tokens)):
                this_tweet_features = perform_function(extract_feature.get_topic_features, corpus, lda_model, index)
                train_features.append({**this_tweet_features})
            test_features = []
            for token, pos in zip(test_tokens, test_pos):
                this_tweet_features = perform_function(extract_feature.get_topic_features_for_unseen_tweet, dictionary,
                                                       lda_model, token.split(), pos.split(), activation[2], activation[3], activation[4])
                test_features.append({**this_tweet_features})

            # Convert feature dictionary to a list of feature values (a dict vectorizer)
            train_features, test_features = utils.extract_features_from_dict(train_features,
                                                                             test_features)

            # Scale the features
            scaled_train_features = utils.feature_scaling(train_features)
            scaled_test_features = utils.feature_scaling(test_features)

            # Get the class ratio
            class_ratio = utils.get_classes_ratio_as_dict(train_labels)

            # Run the models
            utils.run_models(scaled_train_features, train_labels, scaled_test_features, test_labels, class_ratio)


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
            elif feature_set == 'topic':
                this_tweet_features = perform_function(feature_function, token.split(), pos.split())
            features.append({**this_tweet_features})
        return features


def select_active_features(all_features, selected_features):
        # Filter the features to select just the active ones, based on activation
        active_features = []
        for feature in all_features:
            features = {key: feature[key] for key in selected_features}
            active_features.append({**features})
        return active_features


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
    elif feature_set == 'topic':
        features_names = ['topics_no', 'passes', 'use_nouns', 'use_verbs', 'use_all']
        activation_list = \
            [[4, 15, True, True, False], [6, 20, True, True, False], [6, 20, False, False, True],
            [8, 20, True, True, False], [8, 20, False, False, True], [10, 30, False, False, True],
            [15, 30, False, False, True], [20, 40, False, False, True]]
        feature_function = extract_feature.build_lda_model

    # Perform the feature analysis - extract the active features and run the classifiers to gather conclusive results
    perform_feature_analysis(train_tokens, filtered_train_tokens, train_pos, filtered_train_pos, train_labels,
                             test_tokens, filtered_test_tokens, test_pos, filtered_test_pos, test_labels,
                             feature_set, features_names, activation_list, feature_function)


if __name__ == "__main__":
    path = os.getcwd()[:os.getcwd().rfind('/')]
    to_write_filename = path + '/stats/topic_analysis2.txt'
    utils.initialize_writer(to_write_filename)

    train_file = "train.txt"
    test_file = "test.txt"
    word_list = "word_list.txt"

    train_tokens, filtered_train_tokens, train_pos, filtered_train_pos, train_labels, \
    test_tokens, filtered_test_tokens, test_pos, filtered_test_pos, test_labels = \
        data_proc.get_clean_data(train_file, test_file, word_list)

    feature_sets = ['topic'] #['pragmatic', 'sentiment', 'syntactic', 'topic']
    for feature_set in feature_sets:
        print("\nCurrent feature: %s\n" % feature_set)
        print("==================================================================\n")
        build_model(train_tokens, filtered_train_tokens, train_pos, filtered_train_pos, train_labels,
                    test_tokens, filtered_test_tokens, test_pos, filtered_test_pos, test_labels, feature_set)
