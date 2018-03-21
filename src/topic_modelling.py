import os, time, random, utils
import data_processing as data_proc
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import numpy as np


def color_words(model, doc):
    doc = model.id2word.doc2bow(doc)
    doc_topics, word_topics, phi_values = model.get_document_topics(doc, per_word_topics=True)
    topic_colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'orange', 4: 'purple',
                    5: 'pink', 6: 'brown', 7: 'grey', 8: 'navy', 9: 'salmon'}
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    word_pos = 1 / len(doc)     # this is just to make sure that the words are well spaced out
    for word, topics in word_topics:
        ax.text(word_pos, 0.8, model.id2word[word],
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20, color=topic_colors[topics[0]],
                transform=ax.transAxes)
        word_pos += 0.20
    ax.set_axis_off()
    plt.show()


def plot_share_of_topics(path, all_doc_topics, no_random_tweets=10):
    # Obtain 10 random tweets, save their names and topic distributions
    randoms = random.sample(range(len(all_doc_topics)), no_random_tweets)
    doc_names = []
    doc_topics = []
    for rand in randoms:
        doc_topics.append(all_doc_topics[rand])
        doc_names.append("tw %d" % rand)
    doc_topics = np.array(doc_topics)

    # Plot each topic distribution separately
    N, K = doc_topics.shape
    ind = np.arange(N)
    width = 0.5
    for i in range(K):
        plt.bar(ind, doc_topics[:, i], width=width)
        plt.xticks(ind + width / 2, doc_names, size=6)
        plt.title('Share of Topic %d' % i)
        plt.savefig(path + "/plots/topic_modelling/share_of_topic_%d_random_%d_tweets_verbs_and_nouns" % (i, no_random_tweets))
        plt.show()

    # Plot all the topics distributions over the random 10 tweets in a nice bar graph
    plots = []
    height_cumulative = np.zeros(N)

    for k in range(K):
        color = plt.cm.coolwarm(k / K, 1)
        if k == 0:
            p = plt.bar(ind, doc_topics[:, k], width, color=color)
        else:
            p = plt.bar(ind, doc_topics[:, k], width, bottom=height_cumulative, color=color)
        height_cumulative += doc_topics[:, k]
        plots.append(p)
    plt.ylim((0, 1))
    plt.ylabel('Topics')
    plt.title('Topics in random %d tweets' % no_random_tweets)
    plt.xticks(ind + width / 2, doc_names, size=6)
    plt.yticks(np.arange(0, 1, 10))
    topic_labels = ['Topic #{}'.format(k) for k in range(K)]
    plt.legend([p[0] for p in plots], topic_labels, bbox_to_anchor=(1.04, 0.5), loc="center left")
    plt.savefig(path + "/plots/topic_modelling/bar_graph_topic_distribution_random_%d_tweets_verbs_and_nouns" % no_random_tweets, bbox_inches='tight')
    plt.show()

    # Plot a "heat map" of the topics distributions
    plt.pcolor(doc_topics, norm=None, cmap='Reds')
    plt.yticks(np.arange(doc_topics.shape[0]) + 0.5, doc_names)
    plt.xticks(np.arange(doc_topics.shape[1]) + 0.5, topic_labels)
    plt.gca().invert_yaxis()
    plt.xticks(rotation=90)
    plt.colorbar(cmap='Reds')
    plt.tight_layout()
    plt.savefig(path + "/plots/topic_modelling/heat_map_topic_distribution_random_%d_tweets_verbs_and_nouns" % no_random_tweets)
    plt.show()


# Plot top 10 words contributing to each topic
# Every word is plotted in a fontsize proportional to its share associated with its topic
def plot_top_10_words_per_topic(path, ldatopic_words, num_topics=6, num_top_words=10):
    all_probs = [[item[1] for item in topic] for topic in ldatopic_words]
    fontsize_base = 40 / np.max(all_probs)
    for t in range(num_topics):
        plt.subplot(1, num_topics, t + 1)
        plt.ylim(0, num_top_words + 0.5)
        plt.xticks([])
        plt.yticks([])
        words = [item[0] for item in ldatopic_words[t]]
        probs = [item[1] for item in ldatopic_words[t]]
        plt.title('Topic #{}'.format(t))
        for i, (word, share) in enumerate(zip(words, probs)):
            plt.text(0.05, num_top_words - i - 0.5, word, fontsize=fontsize_base * share)
    plt.tight_layout()
    plt.savefig(path + "/plots/topic_modelling/top_10_words_contributing_to_topics_using_verbs_and_nouns")
    plt.show()


def get_documents(tweets, pos_tags, use_nouns=True, use_verbs=False):
    documents = []
    for tweet, pos in zip(tweets, pos_tags):
        documents.append(data_proc.extract_lemmatized_tweet(tweet.split(), pos.split(), use_nouns=use_nouns, use_verbs=use_verbs))
    return documents


def gensim_lda_topic_modelling(path, documents, num_of_topics=6, passes=50, verbose=True, plotTopicsResults=True):
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    if verbose:
        print("Cleaned documents:\n", documents)
        print("\nDictionary:\n", dictionary)
        print("\nCorpus in BoW form: \n", corpus)
    start = time.time()
    ldamodel = LdaModel(corpus=corpus, num_topics=num_of_topics, passes=passes, id2word=dictionary)
    end = time.time()
    print("Completion time for building LDA model: %.3f s = %.3f min" % ((end - start), (end - start) / 60.0))

    ldatopics = ldamodel.show_topics(formatted=False)
    ldatopics_words = [[[word, prob] for word, prob in topic] for topicid, topic in ldatopics]

    if verbose:
        print("\nList of words associated with each topic:\n")
        for i in range(len(ldatopics_words)):
            print("\nTopic %d:\n" % i)
            for w, p in ldatopics_words[i]:
                print(p, " - ", w)

    if plotTopicsResults:
        plot_top_10_words_per_topic(path, ldatopics_words, num_topics=6, num_top_words=10)

    all_documents_topics = [(doc_topics, word_topics, word_phis)
                            for doc_topics, word_topics, word_phis
                            in ldamodel.get_document_topics(corpus, per_word_topics=True)]
    all_doc_topics = []
    for i in range(len(all_documents_topics)):
        doc_topics, word_topics, phi_values = all_documents_topics[i]
        all_doc_topics.append([doc_topics[i][1] for i in range(len(doc_topics))])
        if verbose:
            print('Document topics:', doc_topics)
            print('Word topics:', word_topics)
            print('Phi values:', phi_values)
            print('-------------- \n')

    if plotTopicsResults:
        plot_share_of_topics(path, all_doc_topics, no_random_tweets=10)

    # Plot words coloured differently depending on the topic
    for doc in documents[0:100]:
        if len(doc) > 4:
            color_words(ldamodel, doc)


if __name__ == "__main__":
    path = os.getcwd()[:os.getcwd().rfind('/')]
    filename = "train.txt"
    train_tokens = utils.load_file(path + "/res/tokens/tokens_filtered_clean_original_" + filename)
    train_pos = utils.load_file(path + "/res/pos/pos_filtered_clean_original_" + filename)
    documents = get_documents(train_tokens, train_pos, use_nouns=True, use_verbs=True)
    gensim_lda_topic_modelling(path, documents, num_of_topics=6, passes=50, verbose=False, plotTopicsResults=True)
