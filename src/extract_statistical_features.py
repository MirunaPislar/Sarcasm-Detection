import emoji, re, string, time, os, sys
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import PorterStemmer, ngrams
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import LdaModel
from gensim.corpora import MmCorpus, Dictionary
import data_processing as data_proc


# Get 13 pragmatic features (like the presence of laughter, capitalized words, emojis, hashtags, user mentions)
# but also counts of punctuation, strong affirmatives, negations, intensifiers, tokens and average token size
def get_pragmatic_features(tweet_tokens):
    capitalized_words = laughter = user_mentions = negations = affirmatives \
        = interjections = intensifiers = punctuation = emojis = tweet_len_ch = hashtags = 0
    for t in tweet_tokens:
        tweet_len_ch += len(t)
        if t.isupper():
            capitalized_words = 1   # binary feature marking the presence of capitalized words
        if t.startswith("@"):
            user_mentions = 1   # binary feature for the presence of user mentions
        if t.startswith("#"):
            hashtags += 1       # count-based feature of hashtags used (excluding sarcasm or sarcastic)
        if t.lower().startswith("haha") or re.match('l(o)+l$', t.lower()):
            laughter = 1        # binary feature marking the presence of laughter
        if t in data_proc.strong_negations:
            negations += 1      # count-based feature of strong negations
        if t in data_proc.strong_affirmatives:
            affirmatives += 1   # count-based feature of strong affirmatives
        if t in data_proc.interjections:
            interjections += 1  # count-based feature of relevant interjections
        if t in data_proc.intensifiers:
            intensifiers += 1   # count-based feature of relevant intensifiers
        if t in data_proc.punctuation:
            punctuation += 1    # count-based feature of relevant punctuation signs
        if t in emoji.UNICODE_EMOJI:
            emojis += 1         # count-based feature of relevant punctuation signs
    tweet_len_tokens = len(tweet_tokens)  # get the length of the tweet in tokens
    average_token_length = float(tweet_len_tokens) / max(1.0, float(tweet_len_ch))  # average tweet length
    feature_list = {'tw_len_ch': tweet_len_ch, 'tw_len_tok': tweet_len_tokens, 'avg_len': average_token_length,
                    'capitalized': capitalized_words, 'laughter': laughter, 'user_mentions': user_mentions,
                    'negations': negations, 'affirmatives': affirmatives, 'interjections': interjections,
                    'intensifiers': intensifiers, 'punctuation': punctuation, 'emojis': emojis, 'hashtags':hashtags}
    return feature_list


# Obtain 25 POS Tags from the CMU Twitter Part-of-Speech Tagger according to the paper
# "Part-of-Speech Tagging for Twitter: Annotation, Features, and Experiments" by Gimpel et al.
def get_pos_features(pos_tweet):
    pos_dict = dict.fromkeys(['N', 'O', 'S', '^', 'Z', 'L', 'M', 'V', 'A', 'R', '!', 'D', 'P',
                              '&', 'T', 'X', 'Y', '#', '@', '~', 'U', 'E', '$', ',', 'G'], 0)
    for pos in pos_tweet:
        pos_dict[pos] += 1
    return pos_dict


# Extract the n-grams (specified as a list n = [1, 2, 3, ...])
# e.g if n = [1,2,3] then ngram_features is a dictionary of all unigrams, bigrams and trigrams
# This ngram extractor works for any kind of tokens i.e both words and pos tags
def get_ngrams(tokens, n, use_just_words=False, stem=False, for_semantics=False):
    if len(n) < 1:
        return {}
    if not for_semantics:
        if stem:
            porter = PorterStemmer()
            tokens = [porter.stem(t.lower()) for t in tokens]
        if use_just_words:
            tokens = [t.lower() for t in tokens if not t.startswith('@') and not t.startswith('#')
                      and t not in string.punctuation]
    ngram_tokens = []
    for i in n:
        for gram in ngrams(tokens, i):
            string_token = 'gram '
            for j in range(i):
                string_token += gram[j] + ' '
            ngram_tokens.append(string_token)
    ngram_features = {i: ngram_tokens.count(i) for i in set(ngram_tokens)}
    return ngram_features


# Get sentiment features -- a total of 18 features derived
# Emoji features: a count of the positive, negative and neutral emojis
# along with the ratio of positive to negative emojis and negative to neutral
# Using the MPQA subjectivity lexicon, we have to check words for their part of speech
# and obtain features: a count of positive, negative and neutral words, as well as
# a count of the strong and weak subjectives, along with their ratios and a total sentiment words.
# Also using VADER sentiment analyser to obtain a score of sentiments held in a tweet (4 features)
def get_sentiment_features(tweet, tweet_tokens, tweet_pos, emoji_sent_dict, subj_dict):
    sent_features = dict.fromkeys(["positive emoji", "negative emoji", "neutral emoji",
                                   "emojis pos:neg", "emojis neutral:neg",
                                   "subjlexicon weaksubj", "subjlexicon strongsubj",
                                   "subjlexicon positive", "subjlexicon negative",
                                   "subjlexicon neutral", "words pos:neg", "words neutral:neg",
                                   "subjectivity strong:weak", "total sentiment words"], 0.0)
    for t in tweet_tokens:
        if t in emoji_sent_dict.keys():
            sent_features['negative emoji'] += float(emoji_sent_dict[t][0])
            sent_features['neutral emoji'] += float(emoji_sent_dict[t][1])
            sent_features['positive emoji'] += float(emoji_sent_dict[t][2])

    # Obtain the report of positive to negative emojis
    if sent_features['negative emoji'] != 0:
        if sent_features['positive emoji'] == 0:
            sent_features['emojis pos:neg'] = -1.0 / float(sent_features['negative emoji'])
        else:
            sent_features['emojis pos:neg'] = sent_features['positive emoji'] \
                                            / float(sent_features['negative emoji'])

    # Obtain the report of neutral to negative emojis
    if sent_features['negative emoji'] != 0:
        if sent_features['neutral emoji'] == 0:
            sent_features['emojis neutral:neg'] = -1.0 / float(sent_features['negative emoji'])
        else:
            sent_features['emojis neutral:neg'] = sent_features['neutral emoji'] \
                                            / float(sent_features['negative emoji'])

    lemmatizer = WordNetLemmatizer()
    pos_translation = {'N': 'noun', 'V': 'verb', 'D': 'adj', 'R': 'adverb'}
    for index in range(len(tweet_tokens)):
        stemmed = lemmatizer.lemmatize(tweet_tokens[index], 'v')
        if stemmed in subj_dict.keys():
            if tweet_pos[index] in pos_translation and pos_translation[tweet_pos[index]] in subj_dict[stemmed].keys():
                # Get the type of subjectivity (strong or weak) of this stemmed word
                sent_features['subjlexicon ' + subj_dict[stemmed][pos_translation[tweet_pos[index]]][0]] += 1
                # Get the type of polarity (pos, neg, neutral) of this stemmed word
                if subj_dict[stemmed][pos_translation[tweet_pos[index]]][1] == 'both':
                    sent_features['subjlexicon positive'] += 1
                    sent_features['subjlexicon negative'] += 1
                else:
                    sent_features['subjlexicon ' + subj_dict[stemmed][pos_translation[tweet_pos[index]]][1]] += 1
            else:
                if 'anypos' in subj_dict[stemmed].keys():
                    # Get the type of subjectivity (strong or weak) of this stemmed word
                    sent_features['subjlexicon ' + subj_dict[stemmed]['anypos'][0]] += 1 # strong or weak subjectivity
                    # Get the type of polarity (pos, neg, neutral) of this stemmed word
                    if subj_dict[stemmed]['anypos'][1] == 'both':
                        sent_features['subjlexicon positive'] += 1
                        sent_features['subjlexicon negative'] += 1
                    else:
                        sent_features['subjlexicon ' + subj_dict[stemmed]['anypos'][1]] += 1

    # Use the total number of sentiment words as a feature
    sent_features["total sentiment words"] = sent_features["subjlexicon positive"] \
                                             + sent_features["subjlexicon negative"] \
                                             + sent_features["subjlexicon neutral"]
    if sent_features["subjlexicon negative"] != 0:
        if sent_features["subjlexicon positive"] == 0:
            sent_features["words pos:neg"] = -1.0 / float(sent_features["subjlexicon negative"])
        else:
            sent_features["words pos:neg"] = sent_features["subjlexicon positive"] \
                                           / float(sent_features["subjlexicon negative"])
    if sent_features["subjlexicon negative"] != 0:
        if sent_features["subjlexicon neutral"] == 0:
            sent_features["words neutral:neg"] = -1.0 / float(sent_features["subjlexicon negative"])
        else:
            sent_features["words neutral:neg"] = sent_features["subjlexicon neutral"] \
                                                 / float(sent_features["subjlexicon negative"])
    if sent_features["subjlexicon weaksubj"] != 0:
        if sent_features["subjlexicon strongsubj"] == 0:
            sent_features["subjectivity strong:weak"] = -1.0 / float(sent_features["subjlexicon weaksubj"])
        else:
            sent_features["subjectivity strong:weak"] = sent_features["subjlexicon strongsubj"] \
                                           / float(sent_features["subjlexicon weaksubj"])

    # Vader Sentiment Analyser
    # Obtain the negative, positive, neutral and compound scores of a tweet
    sia = SentimentIntensityAnalyzer()
    polarity_scores = sia.polarity_scores(tweet)
    for name, score in polarity_scores.items():
        sent_features["Vader score " + name] = score
    return sent_features


# Get the necessary data to perform topic modelling
# including clean noun and verb phrases (lemmatized, lower-case)
# Tokenization and POS labelled done as advertised by CMU
def get_topic_data(tokens_tags, pos_tags, path, filename, use_nouns=True, use_verbs=True,
                   num_of_topics=8, passes=25, verbose=True):
    topics_filename = str(num_of_topics) + "topics"
    if use_nouns and use_verbs:
        topics_filename += "_n+v_" + filename
    else:
        topics_filename += "_just_n_" + filename
    lda_filename = path + "/models/lda_" + topics_filename + ".model"
    corpus_filename = path + "res/corpus_" + topics_filename + ".mm"
    if not os.path.exists(lda_filename):
        lemmatizer = WordNetLemmatizer()
        docs = []
        for index in range(len(tokens_tags)):
            tokens = tokens_tags[index].split()
            pos = pos_tags[index].split()
            clean_data = []
            for index2 in range(len(tokens)):
                if use_verbs and pos[index2] is 'V':
                    clean_data.append(lemmatizer.lemmatize(tokens[index2].lower(), 'v'))
                if use_nouns and pos[index2] is 'N':
                    clean_data.append(lemmatizer.lemmatize(tokens[index2].lower()))
            docs.append(clean_data)
        dictionary = Dictionary(docs)
        corpus = [dictionary.doc2bow(d) for d in docs]

        # Save the bow corpus
        MmCorpus.serialize(corpus_filename, corpus)

        if verbose:
            print("\nCleaned documents:")
            print(docs)
            print("\nDictionary:")
            print(dictionary)
            print("\nCorpus in BoW form:")
            print(corpus)

        print("\nStarted building LDA model...")
        start = time.time()
        ldamodel = LdaModel(corpus=corpus, num_topics=num_of_topics, passes=passes, id2word=dictionary)
        ldamodel.save(lda_filename)
        end = time.time()
        print("Completion time for building LDA model: %.3f s = %.3f min" % ((end - start), (end - start) / 60.0))
        if verbose:
            print("\nList of words associated with each topic:")
            ldatopics = ldamodel.show_topics(formatted=False)
            ldatopics_list = [[word for word, prob in topic] for topicid, topic in ldatopics]
            print([t for t in ldatopics_list])
    # Load the previously saved LDA model
    ldamodel = LdaModel.load(lda_filename)
    # Load the previously saved corpus
    mm_corpus = MmCorpus(corpus_filename)
    index = 0
    if verbose:
        for doc_topics, word_topics, word_phis in ldamodel.get_document_topics(mm_corpus, per_word_topics=True):
            print("Index ", index)
            print('Document topics:', doc_topics)
            print('Word topics:', word_topics)
            print('Phi values:', word_phis)
            print('-------------- \n')
            index += 1
    return mm_corpus, ldamodel


# Use the distribution of topics in a tweet as features
def get_topic_features(corpus, ldamodel):
    topic_features = {}
    doc_topics, word_topic, phi_values = ldamodel.get_document_topics(corpus, per_word_topics=True)
    for topic in doc_topics:
        topic_features['topic ' + str(topic[0])] = topic[1]
    return topic_features


# Collect all features
def get_feature_set(tweets, path, filename, pragmatic=False, pos_unigrams=True, pos_bigrams=False,
                    lexical=False, ngram_list=[1], sentiment=False, topic=False):
    features = []
    filename = filename.split("/")[len(filename.split("/")) - 1]

    # Get the pos tags and the tokens of the tweets
    tweets_tokens, tweets_pos = data_proc.get_tags_for_each_tweet(path, filename)

    if sentiment:
        # Emoji lexicon - underlying sentiment (pos, neutral, neg)
        emoji_dict = data_proc.build_emoji_sentiment_dictionary(path)
        # Obtain subjectivity features from the MPQA lexicon and build the subjectivity lexicon
        subj_dict = data_proc.get_subj_lexicon(path)

    if topic:
        corpus, ldamodel = get_topic_data(tweets_tokens, tweets_pos, path, filename[:-4], verbose=False)

    for index in range(0, len(tweets)):
        # print("Tweet %d" % index)
        tokens_this_tweet = tweets_tokens[index].split()
        pos_this_tweet = tweets_pos[index].split()
        pragmatic_features = {}
        pos_unigrams_features = {}
        pos_bigrams_features = {}
        words_ngrams = {}
        sentiment_features = {}
        topic_features = {}
        if pragmatic:
            pragmatic_features = get_pragmatic_features(tokens_this_tweet)
        if pos_unigrams:
            pos_unigrams_features = get_pos_features(pos_this_tweet)
        if pos_bigrams:
            pos_bigrams_features = get_ngrams(pos_this_tweet, n=[3], for_semantics=True)
        if lexical:
            words_ngrams = get_ngrams(tokens_this_tweet, n=ngram_list, use_just_words=True)
        if sentiment:
            sentiment_features = get_sentiment_features(tweets_tokens[index], tokens_this_tweet,
                                                        pos_this_tweet, emoji_dict, subj_dict)
        if topic:
            topic_features = get_topic_features(corpus[index], ldamodel)
        features.append({**pragmatic_features, **pos_unigrams_features, **pos_bigrams_features,
                         **words_ngrams, **sentiment_features, **topic_features})
    return features
