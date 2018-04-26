from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import sys
import os
import logging
import re
import csv
import codecs

import pickle
import numpy as np
from numpy.linalg import norm

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import OrderedDict
from statistics import mean, stdev
from scipy.linalg import norm
from scipy.spatial.distance import euclidean
from scipy.stats import entropy

PERIODS = ["bf", "af"]
_SQRT2 = np.sqrt(2)     # sqrt(2) with default precision np.float64

def read_corpus(event):
    global PERIODS
    corpus_all = {}
    for period in PERIODS:
        file = "reddit_post_v2_{}_{}.csv".format(event, period)
        corpus_pd = pd.read_csv(file, usecols=['text'])
        corpus_all[period] = corpus_pd['text'].values.astype('U').tolist()
        print(corpus_all[period][:10])

    analyze_corpus(corpus_all)
    return corpus_all

def analyze_corpus(corpus_all):
    vocab_size_dist_bf = [len(text.split()) for text in corpus_all[PERIODS[0]]]
    #sns.kdeplot(vocab_size_dist_bf)
    #plt.title("Vocabulary Distribution Over Pre-event Documents ")
    #plt.xlabel("Number of Vocabulary")
    #plt.ylabel("Distribution")
    #plt.show()
    bf_post_num = len(corpus_all[PERIODS[0]])
    af_post_num = len(corpus_all[PERIODS[1]])

    print("len doc", bf_post_num,
          "max_vocab_size", max(vocab_size_dist_bf),
          "mean_vocab_size", mean(vocab_size_dist_bf),
          "stdev_vocab_size", stdev(vocab_size_dist_bf))

    vocab_size_dist_af = [len(text.split()) for text in corpus_all[PERIODS[1]]]
    #sns.kdeplot(vocab_size_dist_af)
    #plt.title("Vocabulary Distribution Over Post-event Documents")
    #plt.xlabel("Number of Vocabulary")
    #plt.ylabel("Distribution")
    #plt.show()
    print("len doc", af_post_num,
          "max_vocab_size", max(vocab_size_dist_af),
          "mean_vocab_size", mean(vocab_size_dist_af),
          "stdev_vocab_size", stdev(vocab_size_dist_af))

    print("Percent increase in post ", (af_post_num - bf_post_num) / bf_post_num)

def find_topics_by_period(corpus_all, n_features, n_topics, incident):
    global PERIODS
    # combine tweets collected before disaster and after disaster
    corpus_t = []
    corpus_t.extend(corpus_all[PERIODS[0]])
    corpus_t.extend(corpus_all[PERIODS[1]])
    [pipeline, model, lda, vect] = find_n_topics(corpus_t, n_features, n_topics, incident)

    num_bf_doc = len(corpus_all[PERIODS[0]])
    compare_topics(pipeline, model, num_bf_doc, incident)

    return lda, vect

def find_n_topics(corpus_t, n_features, n_topics, incident):
    # train the model on the whole data
    override = True
    backup_name = "backup/topic_pipeline_{}_{}.p".format(incident, n_topics)
    if override:
        pipeline = pickle.load(open(backup_name, "rb"))
        model = pipeline.transform(corpus_t)
        lda = pipeline.named_steps['lda']
        vect = pipeline.named_steps['vect']
        return [pipeline, model, lda, vect]

    pipeline = Pipeline([
        ('vect', CountVectorizer(max_df=0.95,
                                 min_df=2,
                                 max_features=n_features,
                                 stop_words='english')),
        ('lda', LatentDirichletAllocation(n_components=n_topics,
                                          max_iter=50,
                                          learning_method='online',
                                          learning_offset=20.,
                                          n_jobs=2)),
    ])

    model = pipeline.fit_transform(corpus_t)
    # save pipeline
    pickle.dump(pipeline, open(backup_name, "wb+"))

    lda = pipeline.named_steps['lda']
    vect = pipeline.named_steps['vect']

    return [pipeline, model, lda, vect]

def get_topic_word_dist(components, feature_names, num_words=100):
    word_dists = []  # [{word1: weight1, word2: weight2, word3: weight3....}, {word1: weight1, word2: weight2....}...]
    for features in components:
        feature_list = OrderedDict()  # {word1: weight, word2: weight ...}
        for i in features.argsort()[:-num_words - 1:-1]:
            feature = feature_names[i]
            weight = features[i]
            feature_list[feature] = weight

        word_dists.append(feature_list)
    return word_dists

def sort_topic_dist(components, feature_names, model, num_topics=20):
    average = np.average(np.array(model), axis=0)
    sorted_topics = average.argsort()[::-1]

    sorted_component = [components[i] for i in sorted_topics]
    topics_words = get_topic_word_dist(sorted_component, feature_names)

    result = {"avg": average, "sorted_topics": sorted_topics, "sorted_component": sorted_component,
              "topics_words": topics_words}
    return result

def write_topics(sorted_topics, topic_words, average, incident, period):
    file_name = "backup/topic_word_dist_{}_{}.csv".format(incident, period)
    with codecs.open(file_name, "w+",'utf-8') as out_file:
        writer = csv.writer(out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for i, topic in enumerate(topic_words):
            topic_str = "topic" + str(sorted_topics[i]) + " | " + str(average[sorted_topics[i]])
            writer.writerow([topic_str, ' | '.join([key + ' ' + str(round(value, 2)) for key, value in topic.items()])])

def distribute_diff(avg1, avg2, type):
    global event
    diff = (avg1 - avg2) / avg1
    #print("diff", diff)

    abs_diff = np.absolute(diff)

    x_coordinate = [i + 1 for i in range(len(abs_diff))]
    plt.plot(x_coordinate, abs_diff * 100.0, 'b')
    plt.xticks(x_coordinate)
    plt.title("Topic distribution percent change")
    plt.xlabel("Topic Number")
    plt.ylabel("Distribution change (%)")

    file_name = "plot_dist_percent_change_{}_{}".format(event, type)
    plt.savefig(file_name)
    plt.close()

    topics = abs_diff.argsort()[::-1]

    topic_diff = {topic: abs_diff[topic] for topic in topics}
    print("topic_diff", topic_diff)

    return topic_diff

def plot_dist(dist1, dist2):
    global event
    x_coordinate = [i + 1 for i in range(len(list(dist1)))]
    plt.plot(x_coordinate, dist1 * 100.0, 'b')
    plt.plot(x_coordinate, dist2 * 100.0, 'g')
    plt.xticks(x_coordinate)
    plt.title("Topic distribution, blue-before, green-after ")
    plt.xlabel("Topic Number")
    plt.ylabel("Distribution")
    file_name = "plot_dist_{}".format(event)
    plt.savefig(file_name)
    plt.close()

def hellinger1(p, q):
    return norm(np.sqrt(p) - np.sqrt(q)) / _SQRT2


def hellinger2(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2


def hellinger3(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2

# zero represent strong relationship
def jsd(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def consine_similarirty(vec1, vec2):
    numerator = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    return numerator / (norm_vec1 * norm_vec2)

def compare_topics(pipeline, model, num_bf_doc, incident):
    components = pipeline._final_estimator.components_
    feature_names = pipeline.named_steps['vect'].get_feature_names()

    print_num = n_topics

    # overall topic distributions
    print("overall topic distribution: \n")
    model_t = model
    result_t = sort_topic_dist(components, feature_names, model_t)
    print("average:", result_t['avg'])
    print("sorted_topics:", [idx for idx in result_t['sorted_topics'][:print_num]])
    for i, words in enumerate([', '.join(list(dist.keys())[:20]) for dist in result_t['topics_words'][:n_topics]]):
        print("topic " + str(result_t['sorted_topics'][i]) + " top words:", words)
    print("\n\n")
    write_topics(result_t['sorted_topics'], result_t['topics_words'], result_t['avg'], incident, 'all')


    print("Before disaster's topic distribution: \n")
    model_bf = model[:num_bf_doc]
    result_bf = sort_topic_dist(components, feature_names, model_bf)
    print("average:", result_bf['avg'])
    print("sorted_topics:", [idx for idx in result_bf['sorted_topics'][:print_num]])
    for i, words in enumerate([', '.join(list(dist.keys())[:20]) for dist in result_bf['topics_words'][:print_num]]):
        print("topic " + str(result_bf['sorted_topics'][i]) + " top words:", words)
    print("\n\n")
    write_topics(result_bf['sorted_topics'], result_bf['topics_words'], result_bf['avg'], incident, 'bf')


    print("After disaster's topic distribution: \n")
    model_af = model[num_bf_doc:]
    result_af = sort_topic_dist(components, feature_names, model_af)
    print("average:", result_af['avg'])
    print("sorted_topics:", [idx for idx in result_af['sorted_topics'][:print_num]])
    for i, words in enumerate([', '.join(list(dist.keys())[:20]) for dist in result_af['topics_words'][:print_num]]):
        print("topic " + str(result_af['sorted_topics'][i]) + " top words:", words)
    print("\n\n")
    write_topics(result_af['sorted_topics'], result_af['topics_words'], result_af['avg'], incident, 'bf')


    print("topic distribution difference: \n")
    topic_diff = distribute_diff(result_bf['avg'], result_af['avg'], "period")
    print("mean between two periods", mean(list(topic_diff.values())), "stdev", stdev(list(topic_diff.values())))

    print("distribution cosine similarity: \n")
    print(consine_similarirty(result_bf['avg'], result_af['avg']))


    print("distribution similarity between bf and af: \n")
    top_num = int(n_topics)
    top_result_bf = [result_bf['avg'][i] for i in list(topic_diff.keys())[:top_num]]
    top_result_af = [result_af['avg'][i] for i in list(topic_diff.keys())[:top_num]]
    print("top_diff_result_bf", top_result_bf)
    print("top_diff_result_af", top_result_af)

    hellinger_between = hellinger1(top_result_bf, top_result_af)
    print("hellinger", hellinger_between)

    jsd_between = jsd(top_result_bf, top_result_af)
    print("JSD", jsd_between)

    #graph distribution
    plot_dist(result_bf['avg'], result_af['avg'])


    print("distribution similarity within bf: \n")
    shuffle_model = np.copy(model)
    np.random.shuffle(shuffle_model)
    print(type(model))
    pivot = int(num_bf_doc/2)
    model_bf_1 = shuffle_model[:pivot]
    result_bf_1 = sort_topic_dist(components, feature_names, model_bf_1)
    model_bf_2 = shuffle_model[pivot:num_bf_doc]
    result_bf_2 = sort_topic_dist(components, feature_names, model_bf_2)
    topic_within_diff = distribute_diff(result_bf_1['avg'], result_bf_2['avg'], "within")
    print("mean within periods", mean(list(topic_within_diff.values())), "stdev", stdev(list(topic_within_diff.values())))

    top_result_bf_1 = [result_bf_1['avg'][i] for i in list(topic_within_diff.keys())[:top_num]]
    top_result_bf_2 = [result_bf_2['avg'][i] for i in list(topic_within_diff.keys())[:top_num]]
    print("top_diff_result_bf1", top_result_bf_1)
    print("top_diff_result_bf2", top_result_bf_2)
    hellinger_within = hellinger1(top_result_bf_1, top_result_bf_2)
    print("hellinger", hellinger_within)

    jsd_witin = jsd(top_result_bf_1, top_result_bf_2)
    print("JSD", jsd_witin)

    print("hellinger distance increased by respect to within the period")
    print(abs(hellinger_between - hellinger_within) / hellinger_within)
    print("jsd_between distance increased by respect to within the period")
    print(abs(jsd_between - jsd_witin) / jsd_witin)


if __name__ == "__main__":
    events = ['NBA_champ']
    n_features = 2500
    n_topics = 20
    for event in events:
        print("         ")
        print("EVENT : ", event)

        corpus_all = read_corpus(event)
        find_topics_by_period(corpus_all, n_features, n_topics, event)
