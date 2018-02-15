import re, os, itertools, string
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import words
import numpy as np
import utils
from vocab_helpers import contractions, implicit_emoticons, slang, wikipedia_emoticons

path = os.getcwd()[:os.getcwd().rfind('/')]


def build_subj_dicionary(lines):
    subj_dict = dict()
    lines = lines.split("\n")
    for line in lines:
        splits = line.split(' ')
        if len(splits) == 6:
            word = splits[2][6:]        # the word analyzed
            word_type = splits[0][5:]   # weak or strong subjective
            pos = splits[3][5:]         # part of speech: noun, verb, adj, adv or anypos
            polarity = splits[5][14:]   # its polarity: can be positive, negative or neutral
            new_dict_entry = {pos: [word_type, polarity]}
            if word in subj_dict.keys():
                subj_dict[word].update(new_dict_entry)
            else:
                subj_dict[word] = new_dict_entry
    return subj_dict


def get_subj_lexicon():
    filename = path + "/res/subjectivity_lexicon.tff"
    lexicon = utils.load_file(filename)
    subj_dict = build_subj_dicionary(lexicon)
    return subj_dict


def get_emoji_dictionary():
    filename = path + "/res/emoji/emoji_list.txt"
    emojis = utils.load_file(filename).split("\n")
    emoji_dict = {}
    for line in emojis:
        line = line.split(" ", 1)
        emoji = line[0]
        description = line[1]
        emoji_dict[emoji] = description
    return emoji_dict


def build_emoji_sentiment_dictionary():
    new_emoji_sentiment_filename = path + "/res/emoji/emoji_sentiment_dictionary.txt"
    if not os.path.exists(new_emoji_sentiment_filename):
        filename = path + "/res/emoji/emoji_sentiment_raw.txt"
        emojis = utils.load_file(filename).split("\n")[1:]
        lines = []
        for line in emojis:
            line = line.split(",")
            emoji = line[0]
            occurences = line[2]
            negative = float(line[4]) / float(occurences)
            neutral = float(line[5]) / float(occurences)
            positive = float(line[6]) / float(occurences)
            description = line[7]
            lines.append(str(emoji) + "\t" + str(negative) + "\t" + str(neutral)
                         + "\t" + str(positive) + "\t" + description.lower())
            utils.save_file(lines, new_emoji_sentiment_filename)
    emoji_sentiment_data = utils.load_file(new_emoji_sentiment_filename).split("\n")
    emoji_sentiment_dict = {}
    for line in emoji_sentiment_data:
        line = line.split("\t")
        # Get emoji characteristics as a list [negative, neutral, positive, description]
        emoji_sentiment_dict[line[0]] = [line[1], line[2], line[3], line[4]]
    return emoji_sentiment_dict


# Replace a contraction (comming from possessives, verbs, emphasis or just bad language) by its longer form
def replace_contracted_form(contracted_word, pos, dictionary):
    long_form = []
    if "'" in contracted_word:
        # print("Found apostrophe in word: ", contracted_word, ' with pos: ', pos)
        split_words = contracted_word.split("'")
        check_if_in_dict = False
        # If the contraction is a nominal + verbal or a proper noun + verbal
        if pos is 'L' or pos is 'M':
            long_form.append(split_words[0])
            if split_words[1].lower() in contractions:
                long_form.extend(contractions[split_words[1].lower()].split())
        # If the contraction is a whole verb (like let's or isn't)
        elif pos in ['V', 'Y', 'O'] and contracted_word.lower() in contractions:
            long_form.extend(contractions[contracted_word.lower()].split())
        # If the contraction is proper noun with possessive or a nominal with a possessive or even a (proper) noun
        elif pos in ['S', 'Z', 'D', 'N', '^']:
            if contracted_word.lower() in contractions:
                long_form.extend(contractions[contracted_word.lower()].split())
            elif split_words[1].lower() == 's':
                long_form.append(split_words[0])
            elif contracted_word.lower() in contractions:
                long_form.extend(contractions[contracted_word.lower()].split())
            else:
                check_if_in_dict = True
        # Can skip ' which are just punctuation marks (usually used to emphasize or quote something)
        elif pos is ',':
            # print("Punctuation, nothing to replace.", split_words[0], ' -- ', split_words[1])
            return []
        # Never replace contractions in emojis or emoticons (will be translated later)
        elif pos is 'E':
            long_form.append(contracted_word)
        else:
            check_if_in_dict = True
        if check_if_in_dict:
            # Attempt to separate words which have been separated by ' by human error
            clean0 = re.findall("[a-zA-Z]+", split_words[0])
            clean1 = re.findall("[a-zA-Z]+", split_words[1])
            if clean0 != [] and clean0[0].lower() in dictionary and clean1 != [] and clean1[0].lower() in dictionary:
                # print("Cleaned to ", clean0, ', ', clean1)
                long_form.extend([clean0[0], clean1[0]])
            else:
                # print("Word couldn't be de-contracted!")
                long_form.append(contracted_word)
        return long_form
    else:
        return long_form.append(contracted_word)


# Cannot do lemmatization with NLTK without changing the case - which we don't want
# So lemmatize but remember if upper case or startign with upper letter
# This will be needed when performing CMU pos-tagging or when extracting pragmatic features
def correct_spelling_but_preserve_case(lemmatizer, word):
    corrected = lemmatizer.lemmatize(word.lower(), 'v')
    corrected = lemmatizer.lemmatize(corrected)
    if word.isupper():
        return corrected.upper()
    if word[0].isupper():
        return corrected[0].upper() + corrected[1:]
    return corrected


# Reduce the length of the pattern (if repeating characters are found)
def reduce_lengthening(word, dictionary):
    if word.lower() in dictionary or word.isnumeric():
        return word
    # Pattern for repeating character sequences of length 2 or greater
    pattern2 = re.compile(r"(.)\1{2,}")
    # Pattern for repeating character sequences of length 1 or greater
    pattern1 = re.compile(r"(.)\1{1,}")
    # Word obtained from stripping repeating sequences of length 2
    word2 = pattern2.sub(r"\1\1", word)
    # Word obtained from stripping repeating sequences of length 1
    word1 = pattern1.sub(r"\1", word)
    # print("Reduced length from ", word, " w2 -- ", word2, " w1 -- ", word1)
    if word1.lower() in dictionary:
        return word1
    else:
        return word2


# Translate emojis (or a group of emojis) into a list of descriptions
def translate_emojis(word, emoji_dict):
    description = []
    emojis = list(word)
    for emoji in emojis:
        if emoji in emoji_dict.keys():
            description.extend(emoji_dict[emoji][3].lower().split())
    if description != []:
        return description
    else:
        return word


# TODO: Numerals - sarcasm heavily relies on them so find a way to extract meaning behind numbers
# TODO: Asterisks - used either to emphasize an idea or to mask offending words (like b*tch or sh*t)
# Attempt to clean each tweet and make it as grammatical as possible
def grammatical_clean(tweets, pos_tags, word_file, filename):
    if not os.path.exists(filename):
        dictionary = utils.load_file(word_file).split()
        emoji_dict = build_emoji_sentiment_dictionary()
        lemmatizer = WordNetLemmatizer()
        corrected_tweets = []
        for tweet, pos_tag in zip(tweets, pos_tags):
            corrected_tweet = []
            # print("Tweet: ", tweet)
            # print("POS: ", pos_tag)
            for word, pos in zip(tweet.split(), pos_tag.split()):
                t = word
                # Remove unnecessary hyphens that just add noise (but not from composed words)
                if t.startswith('-') or t.endswith('-'):
                    t = re.sub('[-]', '', t)
                # Translate emojis (not written with parenthesis, but with symbols)
                emoji_translation = translate_emojis(t, emoji_dict)
                if emoji_translation != t:
                    corrected_tweet.extend(emoji_translation)
                    continue
                # Replace contractions with long-forms
                if "'" in t:
                    long_form = replace_contracted_form(t, pos, dictionary)
                    corrected_tweet.extend(long_form)
                    # print("Removed contracted form of ", t, " to ", long_form)
                    continue
                # Check if token contains repeating characters and if so, remove them
                # Exclude removal of repeating punctuation, numerals, user mentions
                if pos not in [',', '$', '~', '@'] and len(t) > 0:
                    t = correct_spelling_but_preserve_case(lemmatizer, t)
                    reduced = reduce_lengthening(t, dictionary)
                    if reduced != t.lower:
                        # print("Reduced length of word ", t, " to ", reduced)
                        t = reduced
                # Translate emoticons to their description
                if t.lower() in wikipedia_emoticons:
                    translated_emoticon = wikipedia_emoticons[t.lower()].split()
                    # print("WIKI emoticon translated from  ", t, " to ", translated_emoticon)
                    corrected_tweet.extend(translated_emoticon)
                    continue
                # Replace all slang (or twitter abbreviations) to explicit form
                if t.lower() in slang.keys():
                    slang_translation = slang[t.lower()]
                    # print("Slang word replaced from ", t, " to ", slang_translation)
                    corrected_tweet.extend(slang_translation.split())
                    continue
                if t != '':
                    corrected_tweet.append(t)
            corrected_tweets.append(corrected_tweet)
        # Save the grammatical set to filename
        lines = [' '.join(line) for line in corrected_tweets]
        for dirty, corrected in zip(tweets, lines):
            print(dirty)
            print(corrected)
            print()
        # utils.save_file(lines, filename)
        return corrected_tweets
    # Load grammatical set from filename
    corrected_tweets = [[word for word in line.split()] for line in utils.load_file(filename).split("\n")]
    return corrected_tweets


def get_stopwords_list():
    stopwords = utils.load_file(path + "/res/stopwords.txt").split("\n")
    return stopwords


def build_vocabulary(vocab_filename, lines, minimum_occurrence=1):
    if not os.path.exists(vocab_filename):
        print("Building vocabulary...")
        vocabulary = Counter()
        for line in lines:
            vocabulary.update([l.lower() for l in line])
        print("The top 10 most common words: ", vocabulary.most_common(10))
        # Filter all words that appear too rarely or too frequent to be conclusive
        vocabulary = {key: vocabulary[key] for key in vocabulary
                      if vocabulary[key] >= minimum_occurrence
                      and key not in vocabulary.most_common(10)}
        utils.save_file(vocabulary.keys(), vocab_filename)
        print("Vocabulary saved to file \"%s\"" % vocab_filename)
    vocabulary = set(utils.load_file(vocab_filename).split())
    print("Loaded vocabulary of size ", len(vocabulary))
    return vocabulary


def build_vocabulary_for_dnn_tasks(vocab_filename, lines):
    if not os.path.exists(vocab_filename):
        print("Building vocabulary...")
        vocabulary = Counter()
        for line in lines:
            vocabulary.update([l.lower() for l in line])
        vocabulary = {key: vocabulary[key] for key in vocabulary}
        vocabulary = sorted(vocabulary.items(), key=lambda pair: pair[1], reverse=True)
        counter = 1
        indexed_vocabulary = {}
        for (key, _) in vocabulary:
            indexed_vocabulary[key] = counter
            counter += 1
        indexed_vocabulary['unk'] = len(indexed_vocabulary) + 1
        utils.save_dictionary(indexed_vocabulary, vocab_filename)
        print("Vocabulary saved to file \"%s\"" % vocab_filename)
    vocabulary = utils.load_dictionary(vocab_filename)
    print("Loaded vocabulary of size ", len(vocabulary))
    return vocabulary


def vocabulary_filtering(vocabulary, lines):
    filtered_lines = []
    indices = []
    for line in lines:
        filtered_line = []
        individual_word_indices = []
        for word in line:
            word = word.lower()
            if word in vocabulary:
                individual_word_indices.append(vocabulary[word])
                filtered_line.append(word)
            else:
                individual_word_indices.append(vocabulary['unk'])
                filtered_line.append('unk')
        indices.append(individual_word_indices)
        filtered_lines.append(filtered_line)
    return filtered_lines, indices


# Extract the lemmatized nouns and/or verbs from a set of documents - used in LDA modelling
def extract_lemmatized_tweet(tokens, pos, use_verbs=True, use_nouns=True, use_all=False):
    lemmatizer = WordNetLemmatizer()
    clean_data = []
    for index in range(len(tokens)):
        if use_verbs and pos[index] is 'V':
            clean_data.append(lemmatizer.lemmatize(tokens[index].lower(), 'v'))
        if use_nouns and pos[index] is 'N':
            clean_data.append(lemmatizer.lemmatize(tokens[index].lower()))
        if use_all:
            lemmatized_word = lemmatizer.lemmatize(tokens[index].lower(), 'v')
            word = lemmatizer.lemmatize(lemmatized_word)
            if pos[index] not in ['^', ',', '$', '&', '!', '#', '@']:
                clean_data.append(word)
    return clean_data


def ulterior_token_clean(tweets_tokens, vocab_filename, filtered_tokens_filename):
    if not os.path.exists(filtered_tokens_filename):
        lemmatizer = WordNetLemmatizer()
        filtered_tweets_tokens = []
        for tweet_tokens in tweets_tokens:
            filtered_tweet_tokens = []
            for token in tweet_tokens.split():
                filtered_token = lemmatizer.lemmatize(filtered_token, 'v')
                filtered_token = lemmatizer.lemmatize(token.lower())
                filtered_tweet_tokens.append(filtered_token)
            filtered_tweets_tokens.append(filtered_tweet_tokens)
        vocab = build_vocabulary(vocab_filename, filtered_tweets_tokens, minimum_occurrence=5)
        filtered_again = []
        for tweet in filtered_tweets_tokens:
            filtered_again_tweet = []
            for word in tweet:
                if word in vocab:
                    filtered_again_tweet.append(word)
                # else:
                #    filtered_again_tweet.append('unk')
            filtered_again.append(' '.join([f for f in filtered_again_tweet]))
        utils.save_file(filtered_again, filtered_tokens_filename)
    # Load the filtered tokens
    filtered_tweets_tokens = utils.load_file(filtered_tokens_filename).split("\n")
    return filtered_tweets_tokens


def ulterior_pos_clean(tweets_pos_tags, filtered_pos_filename):
    if not os.path.exists(filtered_pos_filename):
        filtered_tweets_pos_tags = []
        for tweet_pos_tag in tweets_pos_tags:
            filtered_tweet = []
            for pos_tag in tweet_pos_tag.split():
                if pos_tag not in ['U', 'G', ',', '@', '~']:
                    filtered_tweet.append(pos_tag)
            filtered_tweets_pos_tags.append(' '.join([f for f in filtered_tweet]))
        utils.save_file(filtered_tweets_pos_tags, filtered_pos_filename)
    # Load the filtered tokens
    filtered_tweets_pos_tags = utils.load_file(filtered_pos_filename).split("\n")
    return filtered_tweets_pos_tags


def get_tags_for_each_tweet(tweets_filename, tokens_filename, pos_filename):
    if not os.path.exists(pos_filename):
        tweets = utils.load_file(tweets_filename).split("\n")
        tokens_lines = []
        pos_lines = []
        tokens_line = ""
        pos_line = ""
        for t in tweets:
            if len(t) < 1:
                tokens_lines.append(tokens_line[:])
                pos_lines.append(pos_line[:])
                tokens_line = ""
                pos_line = ""
            else:
                t_split = t.split("\t")
                tokens_line += t_split[0] + " "
                pos_line += t_split[1] + " "
        utils.save_file(tokens_lines, tokens_filename)
        utils.save_file(pos_lines, pos_filename)
    # Load the tokens and the pos for the tweets in this set
    tokens = utils.load_file(tokens_filename).split("\n")
    pos = utils.load_file(pos_filename).split("\n")
    return tokens, pos


# Split based on camel_case
def camel_case_split(term):
    term = re.sub(r'([0-9]+)', r' \1', term)
    term = re.sub(r'(1st|2nd|3rd|4th|5th|6th|7th|8th|9th|0th)', r'\1 ', term)
    splits = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', term)
    return [s.group(0) for s in splits]


# Split a long, compound hash tag into its component tags. Given the character limit of tweets,
# people would stick words together to save space so this is a useful tool.
# Examples of hash splits from real data (train set) are in /stats/hashtag_splits.txt
# Implementation adapted from https://github.com/matchado/HashTagSplitter
def split_hashtag_to_words_all_possibilities(hashtag, word_dictionary):
    all_possibilities = []
    split_possibility = [hashtag[:i] in word_dictionary for i in reversed(range(len(hashtag) + 1))]
    possible_split_positions = [i for i, x in enumerate(split_possibility) if x is True]

    for split_pos in possible_split_positions:
        split_words = []
        word_1, word_2 = hashtag[:len(hashtag) - split_pos], hashtag[len(hashtag) - split_pos:]

        if word_2 in word_dictionary:
            split_words.append(word_1)
            split_words.append(word_2)
            all_possibilities.append(split_words)
            another_round = split_hashtag_to_words_all_possibilities(word_2, word_dictionary)
            if len(another_round) > 0:
                all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in
                                                         zip([word_1] * len(another_round), another_round)]
        else:
            another_round = split_hashtag_to_words_all_possibilities(word_2, word_dictionary)
            if len(another_round) > 0:
                all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in
                                                         zip([word_1] * len(another_round), another_round)]
    return all_possibilities


def split_hashtag(hashtag, word_list):
    split_words = []
    if hashtag != hashtag.lower() and hashtag != hashtag.upper():
        split_words = camel_case_split(hashtag)
    else:
        j = 0
        while j <= len(hashtag):
            loc = j
            for i in range(j + 1, len(hashtag) + 1, 1):
                if hashtag[j:i].lower() in word_list:
                    loc = i
            if loc == j:
                j += 1
            else:
                split_words.append(hashtag[j:loc])
                j = loc
    split_words = ['#' + str(s) for s in split_words]
    return split_words


# Select the best possible hashtag split based on upper-case
# or component words maximizing the length of the possible word split
def split_hashtag_long_version(hashtag):
    word_file = path + "/res/word_list.txt"
    word_list = utils.load_file(word_file).split()
    word_dictionary = list(set(words.words()))
    for alphabet in "bcdefghjklmnopqrstuvwxyz":
        word_dictionary.remove(alphabet)
    all_poss = split_hashtag_to_words_all_possibilities(hashtag.lower(), word_dictionary)
    max_p = 0
    min_len = 1000
    found = False
    best_p = []
    for poss in all_poss:
        counter = 0
        for p in poss:
            if p in word_list:
                counter += 1
        if counter == len(poss) and min_len > counter:
            found = True
            min_len = counter
            best_p = poss
        else:
            if counter > max_p and not found:
                max_p = counter
                best_p = poss
    best_p_v2 = split_hashtag(hashtag, word_list)
    if best_p != [] and best_p_v2 != []:
        split_words = best_p if len(best_p) < len(best_p_v2) else best_p_v2
    else:
        if best_p == [] and best_p_v2 == []:
            split_words = [hashtag]
        else:
            split_words = best_p if best_p_v2 == [] else best_p_v2
    split_words = ['#' + str(s) for s in split_words]
    return split_words


def split_hashtags2(hashtag, word_list):
    print('Hashtag is ', hashtag)
    # Get rid of the hashtag
    if hashtag.startswith('#'):
        term = hashtag[1:]
    else:
        term = hashtag

    # If the hastag is already an existing word (a single word), return it
    if word_list is not None and term.lower() in word_list:
        return [term.lower()]

    # First, attempt splitting by CamelCase
    if term[1:] != term[1:].lower() and term[1:] != term[1:].upper():
        splits = camel_case_split(term)
    elif '#' in term:
        splits = term.split("#")
    else:
        # Second, build possible splits and choose the best split by assigning
        # a "score" to each possible split, based on the frequency with which a word is occuring
        penalty = -69971
        max_coverage = penalty
        max_splits = 6
        n_splits = 0
        term = re.sub(r'([0-9]+)', r' \1', term)
        term = re.sub(r'(1st|2nd|3rd|4th|5th|6th|7th|8th|9th|0th)', r'\1 ', term)
        term = re.sub(r'([A-Z][^A-Z ]+)', r' \1', term.strip())
        term = re.sub(r'([A-Z]{2,})+', r' \1', term)
        splits = term.strip().split(' ')
        if len(splits) < 3:
            # Splitting lower case and uppercase hashtags in up to 5 words
            chars = [c for c in term.lower()]
            found_all_words = False

            while n_splits < max_splits and not found_all_words:
                for index in itertools.combinations(range(0, len(chars)), n_splits):
                    output = np.split(chars, index)
                    line = [''.join(o) for o in output]
                    score = 0.0
                    for word in line:
                        stripped = word.strip()
                        if stripped in word_list:
                            score += int(word_list.get(stripped))
                        else:
                            if stripped.isnumeric():  # not stripped.isalpha():
                                score += 0.0
                            else:
                                score += penalty
                    score = score / float(len(line))
                    if score > max_coverage:
                        splits = line
                        max_coverage = score
                        line_is_valid_word = [word.strip() in word_list if not word.isnumeric() else True for word in
                                              line]
                        if all(line_is_valid_word):
                            found_all_words = True
                            # print(line, score, line_is_valid_word)
                n_splits = n_splits + 1

    # Used for debugging
    with open(path + '/res/hashtags_dump.txt', 'a') as f:
        if term != '' and len(splits) > 0:
            f.write("Hashtag: %s split to %s" % (hashtag, ' '.join([split for split in splits])))
    print("Split to: ", splits)
    return splits


# Initial tweet cleaning - useful to filter data before tokenization
def clean_tweet(tweet, word_list, split_hashtag_method):
    # Add white space before every punctuation sign so that we can split around it and keep it
    tweet = re.sub('([!?*&%"~`^+{}])', r' \1 ', tweet)
    tweet = re.sub('\s{2,}', ' ', tweet)
    tokens = tweet.split()
    valid_tokens = []
    for word in tokens:
        if word.lower().startswith('#sarca'):           # do not include any #sarca* hashtags
            continue
        if word.startswith('@'):                        # replace all mentions with a general user
            word = '@user'
        if word.startswith('http'):                     # do not include URLs
            continue
        if word.startswith('#'):                        # split hash tag
            word = split_hashtag_method(word[1:], word_list)
            valid_tokens.extend(word)
            continue
        valid_tokens.append(word)
    return ' '.join(valid_tokens)


def process_tweets(tweets, word_list, split_hashtag_method):
    clean_tweets = []
    for tweet in tweets:
        clean_tw = clean_tweet(tweet, word_list, split_hashtag_method)
        clean_tweets.append(clean_tw)
    return clean_tweets


def process_set(dataset_filename, vocab_filename, word_list):
    data, labels = utils.load_data_panda(dataset_filename)
    tweets = process_tweets(data, word_list, split_hashtag)
    vocabulary = build_vocabulary(tweets, vocab_filename, minimum_occurrence=10)
    filtered_tweets = []
    for tweet in tweets:
        filtered_tweets.append([t for t in tweet if t in vocabulary])
    return filtered_tweets, labels


def initial_clean(tweets, clean_filename, word_file, word_file_is_dict=False, split_hashtag_method=split_hashtag):
    if not os.path.exists(clean_filename):
        if word_file_is_dict:
            word_list = utils.load_dictionary(path + "/res/" + word_file)
        else:
            word_list = utils.load_file(path + "/res/" + word_file).split()
        filtered_tweets = process_tweets(tweets, word_list, split_hashtag_method)
        utils.save_file(filtered_tweets, clean_filename)
        return filtered_tweets
    else:
        filtered_tweets = [[word for word in line.split()] for line in utils.load_file(clean_filename).split("\n")]
        return filtered_tweets


def get_clean_data(train_filename, test_filename, word_list):
    tweets_filename = "tweets_"
    tokens_filename = "tokens_"
    pos_filename = "pos_"
    tokens_vocab_filename = "vocabulary_of_tokens_" + train_filename

    # Load the train and test sets
    print("Loading data...")
    train_tweets, train_labels = utils.load_data_panda(path + "/res/datasets/" + train_filename)
    test_tweets, test_labels = utils.load_data_panda(path + "/res/datasets/" + test_filename)

    # Initial clean of data
    initial_clean(train_tweets, path + "/res/datasets/clean_" + train_filename, word_list)
    initial_clean(test_tweets, path + "/res/datasets/clean_" + test_filename, word_list)

    # Get the tags corresponding to the test and train files
    train_tokens, train_pos = \
        get_tags_for_each_tweet(path + "/res/cmu_tweet_tagger/" + tweets_filename + train_filename,
                                path + "/res/tokens/" + tokens_filename + train_filename,
                                path + "/res/pos/" + pos_filename + train_filename)
    test_tokens, test_pos = \
        get_tags_for_each_tweet(path + "/res/cmu_tweet_tagger/" + tweets_filename + test_filename,
                                path + "/res/tokens/" + tokens_filename + test_filename,
                                path + "/res/pos/" + pos_filename + test_filename)
    # Ulterior clean of the tags (tokens and pos)
    
    filtered_train_tokens = \
        ulterior_token_clean(train_tokens, path + "/res/vocabulary/" + tokens_vocab_filename,
                             path + "/res/tokens/filtered_" + tokens_filename + train_filename)
    filtered_test_tokens = \
        ulterior_token_clean(test_tokens, path + "/res/vocabulary/" + tokens_vocab_filename,
                             path + "/res/tokens/filtered_" + tokens_filename + test_filename)
    filtered_train_pos = \
        ulterior_pos_clean(train_pos, path + "/res/pos/filtered_" + pos_filename + train_filename)
    filtered_test_pos = \
        ulterior_pos_clean(test_pos, path + "/res/pos/filtered_" + pos_filename + test_filename)
    
    return train_tokens, filtered_train_tokens, train_pos, filtered_train_pos, train_labels,\
        test_tokens, filtered_test_tokens, test_pos, filtered_test_pos, test_labels


def get_clean_dl_data(train_filename, dev_filename, test_filename, word_list):
    vocab_filename = "dnn_vocabulary_" + train_filename

    # Load the train and test sets
    print("Loading data...")
    train_tweets, train_labels = utils.load_data_panda(path + "/res/datasets/" + train_filename)
    dev_tweets, dev_labels = utils.load_data_panda(path + "/res/datasets/" + dev_filename)
    test_tweets, test_labels = utils.load_data_panda(path + "/res/datasets/" + test_filename)

    # Initial clean of data
    clean_train_tweets = initial_clean(train_tweets, path + "/res/datasets/clean_" + train_filename, word_list)
    clean_dev_tweets = initial_clean(dev_tweets, path + "/res/datasets/clean_" + dev_filename, word_list)
    clean_test_tweets = initial_clean(test_tweets, path + "/res/datasets/clean_" + test_filename, word_list)

    vocabulary = build_vocabulary_for_dnn_tasks(path + "/res/vocabulary/" + vocab_filename, clean_train_tweets)
    clean_train_tweets, train_indices = vocabulary_filtering(vocabulary, clean_train_tweets)
    clean_dev_tweets, dev_indices = vocabulary_filtering(vocabulary, clean_dev_tweets)
    clean_test_tweets, test_indices = vocabulary_filtering(vocabulary, clean_test_tweets)

    return clean_train_tweets, train_indices, train_labels, \
        clean_dev_tweets, dev_indices, dev_labels, \
        clean_test_tweets, test_indices, test_labels, len(vocabulary)


def get_grammatical_data(train_filename, dev_filename, test_filename, dict_list, word_list):
    # Load the train and test sets
    print("Loading data...")
    train_tweets, train_labels = utils.load_data_panda(path + "/res/datasets/" + train_filename)
    dev_tweets, dev_labels = utils.load_data_panda(path + "/res/datasets/" + dev_filename)
    test_tweets, test_labels = utils.load_data_panda(path + "/res/datasets/" + test_filename)

    # Initial clean of data
    initial_clean(train_tweets, path + "/res/datasets/grammatical_" + train_filename, word_list,
                  word_file_is_dict=True, split_hashtag_method=split_hashtags2)
    initial_clean(dev_tweets, path + "/res/datasets/grammatical_" + dev_filename, word_list,
                  word_file_is_dict=True, split_hashtag_method=split_hashtags2)
    initial_clean(test_tweets, path + "/res/datasets/grammatical_" + test_filename, word_list,
                  word_file_is_dict=True, split_hashtag_method=split_hashtags2)

    # Get the tags corresponding to the test and train files
    train_tokens, train_pos = \
        get_tags_for_each_tweet(path + "/res/cmu_tweet_tagger/grammatical_tweets_" + train_filename,
                                path + "/res/tokens/grammatical_tokens_" + train_filename,
                                path + "/res/pos/grammatical_pos_" + train_filename)
    dev_tokens, dev_pos = \
        get_tags_for_each_tweet(path + "/res/cmu_tweet_tagger/grammatical_tweets_" + dev_filename,
                                path + "/res/tokens/grammatical_tokens_" + dev_filename,
                                path + "/res/pos/grammatical_pos_" + dev_filename)
    test_tokens, test_pos = \
        get_tags_for_each_tweet(path + "/res/cmu_tweet_tagger/grammatical_tweets_" + test_filename,
                                path + "/res/tokens/grammatical_tokens_" + test_filename,
                                path + "/res/pos/grammatical_pos_" + test_filename)

    # Clean the data and brind it to the most *grammatical* form possible
    grammatical_train_tweets = grammatical_clean(train_tokens, train_pos, path + "/res/" + dict_list,
                                                 path + "/res/grammatical/finest_grammatical_tweets_" + train_filename)
    grammatical_dev_tweets = grammatical_clean(dev_tokens, dev_pos, path + "/res/" + dict_list,
                                                 path + "/res/grammatical/finest_grammatical_tweets_" + dev_filename)
    grammatical_test_tweets = grammatical_clean(test_tokens, test_pos, path + "/res/" + dict_list,
                                                 path + "/res/grammatical/finest_grammatical_tweets_" + test_filename)
    return grammatical_train_tweets, train_labels, grammatical_dev_tweets, dev_labels,\
        grammatical_test_tweets, test_labels
