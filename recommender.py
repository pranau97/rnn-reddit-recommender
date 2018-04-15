'''
Python script that recommends subreddits to an input username.
'''

import configparser
import pickle
from collections import Counter
import praw
import numpy as np
from tflearn.data_utils import pad_sequences
import tensorflow as tf
from scraper import chunks


def load_labels():
    '''Loads the vocabulary.'''
    labels = []
    with open('model/vocab.dat', 'rb') as infile:
        labels = pickle.load(infile)
    return labels


def collect_user_data(user):
    '''Scrape user's comment history.'''

    # Import configuration parameters, user agent for PRAW Reddit object
    config = configparser.ConfigParser()
    config.read('secrets.ini')

    print("Authenticating...")
    # Load user agent string
    reddit_user_agent = config.get('reddit', 'user_agent')
    client_id = config.get('reddit', 'client_id')
    client_secret = config.get('reddit', 'client_secret')
    password = config.get('reddit', 'password')
    username = config.get('reddit', 'username')

    # Initialize the praw Reddit object
    r = praw.Reddit(user_agent=reddit_user_agent, client_id=client_id,
                    client_secret=client_secret, username=username,
                    password=password)
    praw_user = r.redditor(user)
    user_data = [(user_comment.subreddit.display_name,
                  user_comment.created_utc) for user_comment in praw_user.comments.new(limit=None)]

    # Sort by ascending utc timestamp
    return sorted(user_data, key=lambda x: x[1])


class Recommender():
    '''Recommender class that handles all recommendation activity.'''

    def __init__(self):
        '''Initializations.'''
        self.embedding_weights = np.load('model/load_weights.npy')
        self.labels = load_labels()
        self.graph = None
        self.session = None
        self.input_tensor = None
        self.output_tensor = None
        self.user_subs = None
        self.model = None

    def load_graph(self):
        '''Loads the saved tensorflow model into memory.'''

        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile("model/frozen_model.pb", "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we can use again a convenient built-in function to import a graph_def into the
        # current default Graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="prefix",
                op_dict=None,
                producer_op_list=None
            )

        self.session = tf.Session(graph=graph)
        self.input_tensor = graph.get_tensor_by_name('prefix/InputData/X:0')
        self.output_tensor = graph.get_tensor_by_name(
            "prefix/FullyConnected/Softmax:0")

        return graph

    def user_recs(self, user, n_recs=10):
        '''Display recommendations to the user.'''

        user_data = collect_user_data(user)
        user_sub_seq = [self.labels.index(data[0]) if data[0]
                        in self.labels else 0 for data in user_data]
        non_repeating_subs = []
        chunk_size = 15

        for i, sub in enumerate(user_sub_seq):
            if i == 0:
                non_repeating_subs.append(sub)
            elif sub != user_sub_seq[i-1]:
                non_repeating_subs.append(sub)

        self.user_subs = set([self.labels[sub_index]
                              for sub_index in non_repeating_subs])
        sub_chunks = list(chunks(non_repeating_subs, chunk_size))
        user_input = pad_sequences(sub_chunks,
                                   maxlen=chunk_size, value=0., padding='post')

        if self.graph is None:
            print("Loading model...")
            self.model = self.load_graph()

        print("Evaluating recommendations...")
        sub_probs = self.session.run(self.output_tensor, feed_dict={
            self.input_tensor: user_input
        })
        filtered_probs = [[prob if i not in user_sub_seq else 0 for i,
                           prob in enumerate(prob_list)] for prob_list in sub_probs]
        recs = [np.argmax(probs) for probs in filtered_probs]
        if recs:
            top_x_recs, _cnt = zip(*Counter(recs).most_common(n_recs))
            sub_recs = [self.labels[sub_index] for sub_index in top_x_recs]
        else:
            sub_recs = []

        return sub_recs


if __name__ == '__main__':
    # Load the trained tensorflow model
    REC = Recommender()
    print("Enter username: ")
    USER = input()
    RECS = REC.user_recs(USER)
    print(RECS)
