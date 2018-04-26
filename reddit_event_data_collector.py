from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

import time
import pytz
from datetime import datetime, timedelta

import praw
import pandas as pd
import pickle
import os
import requests
import json

STOP_WORDS = set(stopwords.words('english'))
PERIODS = ["bf", "af"]

def convert_to_utc(local_time):
    pst = pytz.timezone('America/New_York')
    t = datetime(year=local_time[0], month=local_time[1], day=local_time[2])
    t_bf = t - timedelta(days=3)
    t_af = t + timedelta(days=3)

    t = pst.localize(t)
    t_bf = pst.localize(t_bf)
    t_af = pst.localize(t_af)

    print("local t_bf:", t_bf.timetuple(), "local t start:", t.timetuple(), "local t_af:", t_af.timetuple(),)
    utc_time_bf = int(time.mktime(t_bf.timetuple()))
    utc_time = int(time.mktime(t.timetuple()))
    utc_time_af = int(time.mktime(t_af.timetuple()))

    print("utc time:", utc_time_bf, utc_time, utc_time_af)
    return utc_time_bf, utc_time, utc_time_af


def build_event_info():
    events = {}
    # event['event_name'] = [subreddit, [timeline1, timeline2]
    #events['poketmon_go'] = ['gaming', [(1467158400, 1467763200), (1467763200, 1468368000)]] # 07/06/2016 - 07/13/2016
    event_time = convert_to_utc((2018, 2, 4))
    events['superbowl'] = ['sports', [(event_time[0], event_time[1]), (event_time[1], event_time[2])]]


    return events

def get_push_shift_data(after, before, sub):
    url = 'https://api.pushshift.io/reddit/search/submission?&size=1000&after='+str(after)+'&before='+str(before)+'&subreddit='+str(sub)
    print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']

def tokenize_doc(doc):
    global STOP_WORDS

    word_tokens = RegexpTokenizer(r'\w+').tokenize(doc.lower())

    #word_tokens_filtered = [WordNetLemmatizer().lemmatize(w) for w in word_tokens if not w in STOP_WORDS and len(w) > 2]

    word_tokens_filtered = [PorterStemmer().stem(w) for w in word_tokens if not w in STOP_WORDS and len(w) > 2]
    #print("word filtered", word_tokens_filtered)
    doc_filtered = ' '.join(word_tokens_filtered)

    return doc_filtered

def parse_text_using_praw(ids, period):
    reddit = praw.Reddit(client_id='g2shP8ZIe2bdQQ',
                         client_secret='rtryT4t39pXCfXylvzTseUNZn44',
                         password='0917choi',
                         user_agent='PrawTut',
                         username='sehwchoi')

    docs = []
    for i, id in enumerate(ids):
        submission = reddit.submission(id)
        #print("title:", submission.title)

        body = submission.title.strip() + " " + submission.selftext.strip()

        submission.comments.replace_more(limit=0)
        comments = [comment.body.strip() for comment in submission.comments.list()]
        comments_str = ' '.join(comments)

        doc = body + comments_str
        #print("doc:", doc)
        doc_filtered = tokenize_doc(doc)
        if i % 10 == 0:
            print('parsing....', i)
            print('doc', doc_filtered)
        docs.append(doc_filtered)

    docs_pd = pd.DataFrame({'text': docs})
    docs_pd.to_csv("reddit_post_v2_{}_{}.csv".format(key, period))


def collect_data(event, timelines, sub):
    global PERIODS
    for i, timeline in enumerate(timelines):
        post_ids = []
        period = PERIODS[i]
        # Unix timestamp of date to crawl from.
        # 2018/04/01
        after = timeline[0]

        # Unix timestamp of date to crawl to.
        # 2018/04/01
        before = timeline[1]
        data = get_push_shift_data(after, before, sub)

        # Will run until all posts have been gathered
        # from the 'after' date up until 'before' date
        num_post = 0
        while len(data) > 0:
            num_post += len(data)
            for submission in data:
                post_ids.append(submission["id"])
            # Calls getPushshiftData() with the created date of the last submission
            data = get_push_shift_data(sub=sub, after=data[-1]['created_utc'], before=before)

        print("total number of post", num_post)

        obj = {}
        obj['sub'] = sub
        obj['id'] = post_ids
        # Save to json for later use
        with open("submissions_id_{}_{}.json".format(event, period), "w") as jsonFile:
            json.dump(obj, jsonFile)

        parse_text_using_praw(obj['id'], period)


if __name__ =="__main__":
    events = build_event_info()
    for key, value in events.items():
        sub = value[0]
        collect_data(key, value[1], sub)
