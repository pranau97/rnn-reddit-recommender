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


def load_graph(frozen_graph_filename):
    '''Loads the saved tensorflow model into memory.'''

    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
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

    return graph


def collect_user_data(user):
    '''Scrape user's comment history.'''

    # Import configuration parameters, user agent for PRAW Reddit object
    config = configparser.ConfigParser()
    config.read('secrets.ini')

    print("Authenticating...")
    # Load user agent string
    reddit_user_agent = config.get('reddit', 'user_agent')
    client_id = config.get('reddit', 'client_id')
    client_secret = config.get('reddit', 'client_api_key')

    # Initialize the praw Reddit object
    r = praw.Reddit(user_agent=reddit_user_agent,
                    client_id=client_id, client_secret=client_secret)

    print("Getting user history...")
    praw_user = r.redditor(user)
    user_data = [(user_comment.subreddit.display_name,
                  user_comment.created_utc) for user_comment in praw_user.comments.new(limit=None)]

    # Sort by ascending utc timestamp
    return sorted(user_data, key=lambda x: x[1])


def user_recs(user, graph, vocab, n_recs=10, chunk_size=15):
    '''Display recommendations to the user.'''

    user_data = collect_user_data(user)
    user_sub_seq = [vocab.index(data[0]) if data[0]
                    in vocab else 0 for data in user_data]
    non_repeating_subs = []

    for i, sub in enumerate(user_sub_seq):
        if i == 0:
            non_repeating_subs.append(sub)
        elif sub != user_sub_seq[i-1]:
            non_repeating_subs.append(sub)

    _user_subs = set([vocab[sub_index] for sub_index in non_repeating_subs])
    sub_chunks = list(chunks(non_repeating_subs, chunk_size))
    user_input = pad_sequences(sub_chunks,
                               maxlen=chunk_size, value=0., padding='post')

    print("Evaluating recommendations...")
    x = graph.get_tensor_by_name('prefix/InputData/X:0')
    y = graph.get_tensor_by_name("prefix/FullyConnected/Softmax:0")
    with tf.Session(graph=graph) as session:
        sub_probs = session.run(y, feed_dict={
            x: user_input
        })

    # Select the subreddit with highest prediction prob for each of the input subreddit sequences of the user
    recs = [np.argmax(probs) for probs in sub_probs]
    filtered_recs = [
        filt_rec for filt_rec in recs if filt_rec not in user_sub_seq]

    top_x_recs, _cnt = zip(*Counter(filtered_recs).most_common(n_recs))
    sub_recs = [vocab[sub_index] for sub_index in top_x_recs]
    print(sub_recs)


tf.logging.set_verbosity(tf.logging.FATAL)

if __name__ == '__main__':
    # Load the trained tensorflow model
    GRAPH = load_graph("frozen_model.pb")
    with open("vocab.dat", 'rb') as infile:
        VOCAB = pickle.load(infile)
    user_recs('GallowBoob314', GRAPH, VOCAB)
