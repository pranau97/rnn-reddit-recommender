'''
Python script that scrapes subreddits from top user posts in /r/all.
'''

import configparser
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import praw


def scrape_data(n_scrape_loops=10):
    '''Main method to scrape user data using the Reddit API.'''

    # Import configuration parameters, user agent for PRAW Reddit object
    config = configparser.ConfigParser()
    config.read('secrets.ini')
    # load user agent string
    reddit_user_agent = config.get('reddit', 'user_agent')
    client_id = config.get('reddit', 'client_id')
    client_secret = config.get('reddit', 'client_api_key')

    reddit_data = []
    # initialize the praw Reddit object
    r = praw.Reddit(user_agent=reddit_user_agent,
                    client_id=client_id, client_secret=client_secret)
    print("Authenticated as /u/GallowBoob314")

    for scrape_loop in range(n_scrape_loops):
        try:
            all_comments = r.subreddit('all').comments(limit=1000)
            print("Scrape Loop " + str(scrape_loop))
            user_count = 0
            for cmt in all_comments:
                user = cmt.author
                if user:
                    user_count += 1
                    print("Evaluating user: ", user_count, "...", end='\r')
                    for user_comment in user.comments.new(limit=None):
                        reddit_data.append([user.name, user_comment.subreddit.display_name,
                                            user_comment.created_utc])
            print()
        except Exception as e:
            print(e)

    with open('dataset/raw_data.dat', 'wb') as outfile:
        pickle.dump(reddit_data, outfile)

    return reddit_data


def chunks(l, n):
    '''Split the data into chunks.'''
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))


def normalize(lst):
    '''Normalize the data collected.'''
    s = sum(lst)
    normed = [itm/s for itm in lst]
    # Pad last value with what ever difference needed to make sum to exactly 1
    normed[-1] = (normed[-1] + (1 - sum(normed)))
    return normed


def remove_repeating_subs(raw_data):
    '''Remove subreddits that are repeated many times.'''
    cache_data = {}
    prev_usr = None
    past_sub = None
    usr_sub_seq = []

    for comment_data in raw_data:
        current_usr = comment_data[0]

        # New user found in sorted comment data, begin sequence extraction for new user
        if current_usr != prev_usr:

            # Dump sequences to cache for previous user if not in cache
            if prev_usr != None and prev_usr not in cache_data.keys():
                cache_data[prev_usr] = usr_sub_seq

            # Initialize user sub sequence list with first sub for current user
            usr_sub_seq = [comment_data[1]]
            past_sub = comment_data[1]

        else:

            # If still iterating through the same user, add new sub to sequence if not a repeat
            # Check that next sub comment is not a repeat of the last interacted with sub,
            # filtering out repeated interactions
            if comment_data[1] != past_sub:
                usr_sub_seq.append(comment_data[1])
                past_sub = comment_data[1]

        # Update previous user to being the current one before looping to next comment
        prev_usr = current_usr

    return cache_data


def build_training_sequences(usr_data, vocab, vocab_probs, sequence_chunk_size):
    '''Build chronological sequences of interactions with subreddits for every user in the data set.'''
    train_seqs = []

    # Split user subsequences into provided chunks of size sequence_chunk_size
    for _usr, usr_sub_seq in usr_data.items():

        comment_chunks = chunks(usr_sub_seq, sequence_chunk_size)
        for chunk in comment_chunks:

            # For each chunk, filter out potential labels to select as training label,
            # filter by the top subs filter list
            filtered_subs = [vocab.index(sub) for sub in chunk]
            if filtered_subs:

                # Randomly select the label from filtered subs, using the vocab probability distribution to smooth out
                # representation of subreddit labels
                filter_probs = normalize(
                    [vocab_probs[sub_index] for sub_index in filtered_subs])
                label = np.random.choice(filtered_subs, 1, p=filter_probs)[0]
                # Build sequence by ensuring users sub exists in models vocabulary and filtering out the selected
                # label for this subreddit sequence
                chunk_seq = [vocab.index(
                    sub) for sub in chunk if sub in vocab and vocab.index(sub) != label]
                train_seqs.append([chunk_seq, label, len(chunk_seq)])

    return train_seqs


def process_data(raw_data):
    '''Loads the collected data as a Pandas dataframe and begins the data cleaning process.'''

    df = pd.DataFrame(raw_data, columns=['user', 'subreddit', 'utc_stamp'])

    train_data = None

    vocab_counts = df["subreddit"].value_counts()
    tmp_vocab = list(vocab_counts.keys())
    total_counts = sum(vocab_counts.values)
    inv_prob = [total_counts/vocab_counts[sub] for sub in tmp_vocab]

    # Build placeholder, 'Unseen-Sub', for all subs not in vocabulary
    vocab = ["Unseen-Sub"] + tmp_vocab
    tmp_vocab_probs = normalize(inv_prob)

    # Force probability sum to 1 by adding differenc to "Unseen-sub" probability
    vocab_probs = [1 - sum(tmp_vocab_probs)] + tmp_vocab_probs
    print("Vocab size = " + str(len(vocab)))
    with open("vocab.dat", 'wb') as outfile:
        pickle.dump(vocab, outfile)

    pp_user_data = remove_repeating_subs(raw_data)
    sequence_chunk_size = 15
    train_data = build_training_sequences(
        pp_user_data, vocab, vocab_probs, sequence_chunk_size)
    sequences, labels, lengths = zip(*train_data)
    train_df = pd.DataFrame({'sub_seqs': sequences,
                             'sub_label': labels,
                             'seq_length': lengths})
    print(train_df.head())

    train_df.to_csv('final_data.csv', sep='\t', encoding='utf-8')


if __name__ == '__main__':
    # Check if data has already been scraped
    DATA = Path("dataset/raw_data.dat")
    if DATA.is_file():
        with open('dataset/raw_data.dat', 'rb') as infile:
            DATA = pickle.load(infile)
    else:
        DATA = scrape_data(10)
    print("Number of user records processed: ", len(DATA))
    process_data(DATA)
