'''
Python script that uses Tensorflow to train an LSTM RNN on subreddit data.
'''

# pylint: disable-msg=E0611

import random
import ast
import csv
import pickle
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.python.framework import graph_util
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, LabelSet


def train_model(train, test, vocab_size, n_epoch=5, n_units=128):
    '''Method to load the dataset and train the RNN model.'''

    train_x = train['sub_seqs']
    train_y = train['sub_label']
    test_x = test['sub_seqs']
    test_y = test['sub_label']
    sequence_chunk_size = 15
    learning_rate = 0.0001
    dropout = 0.6

    # Sequence padding
    train_x = pad_sequences(
        train_x, maxlen=sequence_chunk_size, value=0., padding='post')
    test_x = pad_sequences(test_x, maxlen=sequence_chunk_size,
                           value=0., padding='post')

    # Converting labels to binary vectors
    train_y = to_categorical(train_y, nb_classes=vocab_size)
    test_y = to_categorical(test_y, nb_classes=vocab_size)

    print("Building network topology...")
    # Network building
    net = tflearn.input_data([None, 15])
    net = tflearn.embedding(net, input_dim=vocab_size,
                            output_dim=128, trainable=True)
    net = tflearn.gru(net, n_units=n_units, dropout=dropout,
                      weights_init=tflearn.initializations.xavier(), return_seq=False)
    net = tflearn.fully_connected(
        net, vocab_size, activation='softmax', weights_init=tflearn.initializations.xavier())
    net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=2)

    print("Training model...")
    model.fit(train_x, train_y, validation_set=(test_x, test_y), show_metric=False,
              batch_size=256, n_epoch=n_epoch)

    # For visualizations
    embedding = tflearn.get_layer_variables_by_name("Embedding")[0]

    return [model, embedding]


def visualize_model(model, embedding, vocab):
    '''Helps visualize the trained TF model using BokehJS.'''
    final_weights = model.get_weights(embedding)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    load_weights = tsne.fit_transform(final_weights)

    # Control the number of labelled subreddits to display
    sparse_labels = [lbl if random.random() <= 0.01 else '' for lbl in vocab]
    source = ColumnDataSource(
        {'x': load_weights[:, 0], 'y': load_weights[:, 1], 'labels': sparse_labels})

    tools = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"

    p = figure(tools=tools)

    p.scatter("x", "y", radius=0.1, fill_alpha=0.6,
              line_color=None, source=source)

    labels = LabelSet(x="x", y="y", text="labels", y_offset=8,
                      text_font_size="10pt", text_color="#555555", text_align='center',
                      source=source)
    p.add_layout(labels)

    output_file("visualization.html")
    show(p)


def train_test_split(train_df, vocab):
    '''Create training and testing data by splitting the model in 80:20 ratio.'''

    split_perc = 0.8
    print("Splitting training and testing data...")
    train_len, test_len = np.floor(
        len(train_df) * split_perc), np.floor(len(train_df) * (1-split_perc))
    train, test = train_df.ix[:train_len -
                              1], train_df.ix[train_len:train_len + test_len]
    [model, embedding] = train_model(train, test, vocab)
    return [model, embedding]


def freeze_graph(model):
    '''Saves the model for usage in applications.'''

    # We specify the file fullname of our freezed graph
    output_graph = "frozen_model.pb"

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes
    output_node_names = "InputData/X,FullyConnected/Softmax"

    # We import the meta graph and retrieve a Saver

    # We retrieve the protobuf graph definition
    graph = model.net.graph
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    # We use a built-in TF helper to export variables to constants

    session = model.session
    output_graph_def = graph_util.convert_variables_to_constants(
        session,  # The session is used to retrieve the weights
        input_graph_def,  # The graph_def is used to retrieve the nodes
        # The output node names are used to select the usefull nodes
        output_node_names.split(",")
    )

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as outfile:
        outfile.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))


def load_data():
    '''Loads the cleaned data into a Pandas dataframe.'''

    with open("final_data.csv", 'r') as csv_in:
        csv_in = csv.reader(csv_in, delimiter='\t')
        data = pd.DataFrame(columns=['seq_length', 'sub_label', 'sub_seqs'])
        row_count = 0
        for row in csv_in:
            if row_count == 0:
                row_count += 1
                continue

            seq_list = ast.literal_eval(row[3])
            new_row = {'seq_length': int(row[1]),
                       'sub_label': int(row[2]),
                       'sub_seqs': seq_list}
            data = data.append(new_row, ignore_index=True)

    print(data.head())
    print("Successfully loaded data")

    return data


tf.logging.set_verbosity(tf.logging.FATAL)

if __name__ == '__main__':
    # Load cleaned data into memory
    DATA = load_data()
    with open("vocab.dat", 'rb') as infile:
        VOCAB = pickle.load(infile)

    # Begin training process
    MODEL, EMBEDDING = train_test_split(DATA, len(VOCAB))

    # Freeze the model
    freeze_graph(MODEL)

    # Visualize the model
    visualize_model(MODEL, EMBEDDING, VOCAB)
