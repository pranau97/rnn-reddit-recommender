'''Web server that serves the recommender app.'''

import os
import random
from flask import Flask, request, render_template
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.models import HoverTool
from bokeh.embed import components
from recommender import Recommender

APP = Flask(__name__)

REC = Recommender()


@APP.route("/")
def index():
    '''Renderer for the root webpage.'''
    sparse_labels = [lbl if random.random(
    ) <= 0.01 else '' for lbl in REC.labels]
    source = ColumnDataSource(
        {'x': REC.embedding_weights[:, 0], 'y': REC.embedding_weights[:, 1], 'labels': REC.labels, 'sparse_labels': sparse_labels})
    hover = HoverTool(
        tooltips="""
                      <div>
                        <span>@labels</span>
                      </div>
                      """)

    tools = [hover, "crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"]

    p = figure(tools=tools)
    p.sizing_mode = 'scale_width'

    p.scatter("x", "y", radius=0.1, fill_alpha=0.6,
              fill_color="#c7e9b4",
              line_color=None, source=source)

    labels = LabelSet(x="x", y="y", text="sparse_labels", y_offset=8,
                      text_font_size="8pt", text_color="#555555", text_align='center',
                      source=source)
    p.add_layout(labels)
    script, div = components(p)
    return render_template("index.html", script=script, div=div)


@APP.route("/all")
def all_pages():
    '''Renderer for /all.'''
    source = ColumnDataSource(
        {'x': REC.embedding_weights[:, 0], 'y': REC.embedding_weights[:, 1], 'labels': REC.labels, 'sparse_labels': REC.labels})
    hover = HoverTool(
        tooltips="""
                      <div>
                        <span>@labels</span>
                      </div>
                      """)

    tools = [hover, "crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"]

    p = figure(tools=tools)
    p.sizing_mode = 'scale_width'

    p.scatter("x", "y", radius=0.1, fill_alpha=0.6,
              fill_color="#c7e9b4",
              line_color=None, source=source)

    labels = LabelSet(x="x", y="y", text="sparse_labels", y_offset=8,
                      text_font_size="8pt", text_color="#555555", text_align='center',
                      source=source)
    p.add_layout(labels)
    script, div = components(p)
    return render_template("all.html", script=script, div=div)


@APP.route("/recommend")
def recommend():
    '''Renderer for /recommend.'''
    # initial call to runplan which displays the planner simulation
    user = request.args.get('user')
    if len(user) <= 1:
        return 'Please input a username'
    recommendations = REC.user_recs(user)
    flag = ''
    if recommendations == []:
        flag = "Not enough comment history to provide recommendations"
    colors = []
    for label in REC.labels:
        if label in recommendations:
            colors.append("#0c2c84")
        elif label in REC.user_subs:
            colors.append("#7fcdbb")
        else:
            colors.append("#c7e9b4")

    sparse_labels = [
        lbl if lbl in recommendations or lbl in REC.user_subs else '' for lbl in REC.labels]
    source = ColumnDataSource({'x': REC.embedding_weights[:, 0], 'y': REC.embedding_weights[:, 1],
                               'labels': REC.labels, 'sparse_labels': sparse_labels, 'colors': colors})
    hover = HoverTool(
        tooltips="""
            <div>
                <span>@labels</span>
            </div>
            """
    )

    tools = [hover, "crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"]

    p = figure(tools=tools)
    p.sizing_mode = 'scale_width'

    p.scatter("x", "y", radius=0.1, fill_alpha=0.6,
              fill_color="colors",
              line_color=None, source=source)

    labels = LabelSet(x="x", y="y", text="sparse_labels", y_offset=8,
                      text_font_size="8pt", text_color="#555555", text_align='center',
                      source=source)
    p.add_layout(labels)
    script, div = components(p)

    return render_template("recommend.html", recommendations=recommendations, script=script, div=div, flag=flag)


if __name__ == "__main__":

    PORT = 8000
    # Open a web browser pointing at the app.
    os.system("open http://localhost:{0}/".format(PORT))
    # Set up the development server on port 8000.
    APP.debug = False
    APP.run()
