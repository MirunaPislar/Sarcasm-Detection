import emoji, re, time, os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
from nltk import ngrams
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import LdaModel
from gensim.corpora import MmCorpus, Dictionary
import data_processing as data_proc
import utils
import vocab_helpers as helper
from tqdm import tqdm


# Obtain 6 pragmatic features
def get_pragmatic_features(tweet_tokens):
    capitalized_words = user_specific = intensifiers = tweet_len_ch = 0
    for t in tweet_tokens:
        tweet_len_ch += len(t)
        if t.isupper() and len(t) > 1:
            capitalized_words += 1       # count of capitalized words
        if t.startswith("@"):
            user_specific += 1          # count of user mentions
        if t.startswith("#"):
            user_specific += 1          # count-based feature of hashtags used (excluding sarcasm or sarcastic)
        if t.lower().startswith("haha") or re.match('l(o)+l$', t.lower()):
            user_specific += 1          # binary feature marking the presence of laughter
        if t in helper.strong_negations:
            intensifiers += 1           # count-based feature of strong negations
        if t in helper.strong_affirmatives:
            intensifiers += 1           # count-based feature of strong affirmatives
        if t in helper.interjections:
            intensifiers += 1           # count-based feature of relevant interjections
        if t in helper.intensifiers:
            intensifiers += 1           # count-based feature of relevant intensifiers
        if t in helper.punctuation:
            user_specific += 1          # count-based feature of relevant punctuation signs
        if t in emoji.UNICODE_EMOJI:
            user_specific += 1          # count-based feature of emojis
    tweet_len_tokens = len(tweet_tokens)  # get the length of the tweet in tokens
    average_token_length = float(tweet_len_tokens) / max(1.0, float(tweet_len_ch))  # average tweet length
    feature_list = {'tw_len_ch': tweet_len_ch, 'tw_len_tok': tweet_len_tokens, 'avg_len': average_token_length,
                    'capitalized': capitalized_words, 'user_specific': user_specific, 'intensifiers': intensifiers}
    return feature_list


# Extract the n-grams (specified as a list n = [1, 2, 3, ...])
# e.g if n = [1,2,3] then n-gram_features is a dictionary of all uni-grams, bi-grams and tri-grams
# This n-gram extractor works for any kind of tokens i.e both words and pos tags
def get_ngrams(tokens, n, syntactic_data=False):
    if len(n) < 1:
        return {}
    if not syntactic_data:
        filtered = []
        stopwords = data_proc.get_stopwords_list()
        for t in tokens:
            if t not in stopwords and t.isalnum():
                filtered.append(t)
        tokens = filtered
    ngram_tokens = []
    for i in n:
        for gram in ngrams(tokens, i):
            string_token = str(i) + '-gram '
            for j in range(i):
                string_token += gram[j] + ' '
            ngram_tokens.append(string_token)
    ngram_features = {i: ngram_tokens.count(i) for i in set(ngram_tokens)}
    return ngram_features


# Get sentiment features -- a total of 16 features derived
# Emoji features: a count of the positive, negative and neutral emojis
# along with the ratio of positive to negative emojis and negative to neutral
# Using the MPQA subjectivity lexicon, we have to check words for their part of speech
# and obtain features: a count of positive, negative and neutral words, as well as
# a count of the strong and weak subjectives, along with their ratios and a total sentiment words.
# Also using VADER sentiment analyser to obtain a score of sentiments held in a tweet (4 features)
def get_sentiment_features(tweet, tweet_tokens, tweet_pos, emoji_sent_dict, subj_dict):
    sent_features = dict.fromkeys(["positive emoji", "negative emoji", "neutral emoji",
                                   "subjlexicon weaksubj", "subjlexicon strongsubj",
                                   "subjlexicon positive", "subjlexicon negative",
                                   "subjlexicon neutral", "total sentiment words",
                                   "swn pos", "swn neg", "swn obj"], 0.0)
    for t in tweet_tokens:
        if t in emoji_sent_dict.keys():
            sent_features['negative emoji'] += float(emoji_sent_dict[t][0])
            sent_features['neutral emoji'] += float(emoji_sent_dict[t][1])
            sent_features['positive emoji'] += float(emoji_sent_dict[t][2])

    lemmatizer = WordNetLemmatizer()
    pos_translation = {'N': 'noun', 'V': 'verb', 'D': 'adj', 'R': 'adverb'}
    for index in range(len(tweet_tokens)):
        lemmatized = lemmatizer.lemmatize(tweet_tokens[index], 'v')
        if lemmatized in subj_dict.keys():
            if tweet_pos[index] in pos_translation and pos_translation[tweet_pos[index]] in subj_dict[lemmatized].keys():
                # Get the type of subjectivity (strong or weak) of this lemmatized word
                sent_features['subjlexicon ' + subj_dict[lemmatized][pos_translation[tweet_pos[index]]][0]] += 1
                # Get the type of polarity (pos, neg, neutral) of this lemmatized word
                if subj_dict[lemmatized][pos_translation[tweet_pos[index]]][1] == 'both':
                    sent_features['subjlexicon positive'] += 1
                    sent_features['subjlexicon negative'] += 1
                else:
                    sent_features['subjlexicon ' + subj_dict[lemmatized][pos_translation[tweet_pos[index]]][1]] += 1
            else:
                if 'anypos' in subj_dict[lemmatized].keys():
                    # Get the type of subjectivity (strong or weak) of this lemmatized word
                    sent_features['subjlexicon ' + subj_dict[lemmatized]['anypos'][0]] += 1 # strong or weak subjectivity
                    # Get the type of polarity (pos, neg, neutral) of this lemmatized word
                    if subj_dict[lemmatized]['anypos'][1] == 'both':
                        sent_features['subjlexicon positive'] += 1
                        sent_features['subjlexicon negative'] += 1
                    else:
                        sent_features['subjlexicon ' + subj_dict[lemmatized]['anypos'][1]] += 1

    # Use the total number of sentiment words as a feature
    sent_features["total sentiment words"] = sent_features["subjlexicon positive"] \
                                             + sent_features["subjlexicon negative"] \
                                             + sent_features["subjlexicon neutral"]

    # Obtain average of all sentiment words (pos, ne, obj) using SentiWordNet Interface
    pos_translation = {'N': 'n', 'V': 'v', 'D': 'a', 'R': 'r'}
    for index in range(len(tweet_tokens)):
        lemmatized = lemmatizer.lemmatize(tweet_tokens[index], 'v')
        if tweet_pos[index] in pos_translation:
            synsets = list(swn.senti_synsets(lemmatized, pos_translation[tweet_pos[index]]))
            pos_score = 0
            neg_score = 0
            obj_score = 0
            if len(synsets) > 0:
                for syn in synsets:
                    pos_score += syn.pos_score()
                    neg_score += syn.neg_score()
                    obj_score += syn.obj_score()
                sent_features["swn pos"] = pos_score / float(len(synsets))
                sent_features["swn neg"] = neg_score / float(len(synsets))
                sent_features["swn obj"] = obj_score / float(len(synsets))

    # Vader Sentiment Analyser
    # Obtain the negative, positive, neutral and compound scores of a tweet
    sia = SentimentIntensityAnalyzer()
    polarity_scores = sia.polarity_scores(tweet)
    for name, score in polarity_scores.items():
        sent_features["Vader score " + name] = score
    return sent_features


# Get the necessary data to perform topic modelling, including clean noun and verb phrases (lemmatized, lower-case)
# Tokenization and POS labelled done as advertised by CMU Tweet POS Tagger
def build_lda_model(tokens_tags, pos_tags, use_nouns=True, use_verbs=True, use_all=False,
                    num_of_topics=8, passes=25, verbose=True):
    path = os.getcwd()[:os.getcwd().rfind('/')]
    topics_filename = str(num_of_topics) + "topics"
    if use_nouns:
        topics_filename += "_nouns"
    if use_verbs:
        topics_filename += "_verbs"
    if use_all:
        topics_filename += "_all"

    # Set the LDA, Dictionary and Corpus filenames
    lda_filename = path + "/models/topic_models/lda_" + topics_filename + ".model"
    dict_filename = path + "/res/topic_data/dict/dict_" + topics_filename + ".dict"
    corpus_filename = path + "/res/topic_data/corpus/corpus_" + topics_filename + ".mm"

    # Build a topic model if it wasn't created yet
    if not os.path.exists(lda_filename):
        # Extract the lemmatized documents
        docs = []
        for index in range(len(tokens_tags)):
            tokens = tokens_tags[index].split()
            pos = pos_tags[index].split()
            docs.append(data_proc.extract_lemmatized_tweet(tokens, pos, use_verbs, use_nouns, use_all))

        # Compute the dictionary and save it
        dictionary = Dictionary(docs)
        dictionary.filter_extremes(keep_n=40000)
        dictionary.compactify()
        Dictionary.save(dictionary, dict_filename)

        # Compute the bow corpus and save it
        corpus = [dictionary.doc2bow(d) for d in docs]
        MmCorpus.serialize(corpus_filename, corpus)

        if verbose:
            print("\nCleaned documents:", docs)
            print("\nDictionary:", dictionary)
            print("\nCorpus in BoW form:", corpus)

        # Start training an LDA Model
        start = time.time()
        print("\nBuilding the LDA topic model...")
        lda_model = LdaModel(corpus=corpus, num_topics=num_of_topics, passes=passes, id2word=dictionary)
        lda_model.save(lda_filename)
        end = time.time()
        print("Completion time for building LDA model: %.3f s = %.3f min" % ((end - start), (end - start) / 60.0))

        if verbose:
            print("\nList of words associated with each topic:")
            lda_topics = lda_model.show_topics(formatted=False)
            lda_topics_list = [[word for word, prob in topic] for topic_id, topic in lda_topics]
            print([t for t in lda_topics_list])

    # Load the previously saved dictionary
    dictionary = Dictionary.load(dict_filename)

    # Load the previously saved corpus
    mm_corpus = MmCorpus(corpus_filename)

    # Load the previously saved LDA model
    lda_model = LdaModel.load(lda_filename)

    # Print the top 10 words for each topic
    if verbose:
        for topic_id in range(num_of_topics):
            print("\nTop 10 words for topic ", topic_id)
            print([dictionary[word_id] for (word_id, prob) in lda_model.get_topic_terms(topic_id, topn=10)])

    index = 0
    if verbose:
        for doc_topics, word_topics, word_phis in lda_model.get_document_topics(mm_corpus, per_word_topics=True):
            print('Index ', index)
            print('Document topics:', doc_topics)
            print('Word topics:', word_topics)
            print('Phi values:', word_phis)
            print('-------------- \n')
            index += 1
    return dictionary, mm_corpus, lda_model


# Predict the topic of an unseen testing example based on the LDA model built on the train set
def get_topic_features_for_unseen_tweet(dictionary, lda_model, tokens_tags, pos_tags,
                                        use_nouns=True, use_verbs=True, use_all=False):
    # Extract the lemmatized documents
    docs = data_proc.extract_lemmatized_tweet(tokens_tags, pos_tags, use_verbs, use_nouns, use_all)
    tweet_bow = dictionary.doc2bow(docs)
    topic_prediction = lda_model[tweet_bow]
    topic_features = {}
    if any(isinstance(topic_list, type([])) for topic_list in topic_prediction):
        topic_prediction = topic_prediction[0]
    for topic in topic_prediction:
        topic_features['topic ' + str(topic[0])] = topic[1]
    return topic_features


# Use the distributions of topics in a tweet as features
def get_topic_features(corpus, ldamodel, index):
    topic_features = {}
    doc_topics, word_topic, phi_values = ldamodel.get_document_topics(corpus, per_word_topics=True)[index]
    for topic in doc_topics:
        topic_features['topic ' + str(topic[0])] = topic[1]
    return topic_features


# Get the most similar and the most disimilar scores of a pair of words in a tweet (based on an embedding vector map)
def get_similarity_scores(tweet, vec_map, weighted=True):
    most_similar, most_dissimilar = utils.get_similarity_measures(tweet, vec_map, weighted=weighted, verbose=False)
    return {'most similar ': most_similar, 'most dissimilar ': most_dissimilar}


# Collect all features
def get_feature_set(tweets_tokens, tweets_pos, pragmatic=True, lexical=True,
                    ngram_list=[1], pos_grams=True, pos_ngram_list=[1, 2],
                    sentiment=True, topic=True, similarity=True, word2vec_map=None):
    pragmatic_features = []
    pos_grams_features = []
    words_ngrams = []
    sentiment_features = []
    topic_features = []
    similarity_features = []

    if sentiment:
        # Load the emoji lexicon to get the underlying emoji sentiments (pos, neutral, neg)
        emoji_dict = data_proc.build_emoji_sentiment_dictionary()
        # Obtain subjectivity features from the MPQA lexicon and build the subjectivity lexicon
        subj_dict = data_proc.get_subj_lexicon()

    if topic:
        use_nouns = True
        use_verbs = True
        use_all = False
        dictionary, corpus, ldamodel = build_lda_model(tweets_tokens, tweets_pos,
                                                       use_nouns=use_nouns, use_verbs=use_verbs, use_all=use_all,
                                                       num_of_topics=8, passes=20, verbose=False)

    for index in tqdm(range(len(tweets_tokens))):
        tokens_this_tweet = tweets_tokens[index].split()
        pos_this_tweet = tweets_pos[index].split()
        if pragmatic:
            pragmatic_features.append(get_pragmatic_features(tokens_this_tweet))
            if lexical:
                words_ngrams.append(get_ngrams(tokens_this_tweet, n=ngram_list, syntactic_data=False))
        if pos_grams:
            pos_grams_features.append(get_ngrams(pos_this_tweet, n=pos_ngram_list, syntactic_data=True))
        if sentiment:
            sentiment_features.append(get_sentiment_features(tweets_tokens[index], tokens_this_tweet,
                                                             pos_this_tweet, emoji_dict, subj_dict))
        if topic:
            topic_features.append(get_topic_features_for_unseen_tweet(dictionary, ldamodel, tokens_this_tweet,
                                  pos_this_tweet, use_nouns=use_nouns, use_verbs=use_verbs, use_all=use_all))
        if similarity:
            similarity_features.append(get_similarity_scores(tweets_tokens[index], word2vec_map, weighted=True))

    # Return all features individually
    return pragmatic_features, words_ngrams, pos_grams_features, sentiment_features, topic_features, similarity_features
