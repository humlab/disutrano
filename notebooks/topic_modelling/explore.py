# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Text Analysis - Topic Modeling
# ### <span style='color: green'>SETUP </span> Prepare Notebook and Load Model <span style='float: right; color: red'>MANDATORY</span>

# %%

import __paths__  # pylint: disable=unused-import
from typing import Callable

import bokeh.plotting
import penelope.notebook.topic_modelling as ntm
from IPython.display import display
from penelope.utility import pandas_utils

from notebooks.source.state_on_load import assign_pivot_keys_on_load

bokeh.plotting.output_notebook(hide_banner=True)
pandas_utils.set_default_options()

corpus_folder: str = __paths__.data_folder

current_state: Callable[[], ntm.TopicModelContainer] = ntm.TopicModelContainer.singleton

current_state().register(None, callback=assign_pivot_keys_on_load)

# %% [markdown]
# ### <span style='color: green'>PREPARE</span> Load Topic Model <span style='float: right; color: red'>MANDATORY</span>

# %%
load_gui: ntm.LoadGUI = ntm.LoadGUI(data_folder=corpus_folder, state=current_state()).setup()
display(load_gui.layout())
# %% [markdown]
# ### <span style='color: green'>PREPARE </span> Edit Topic Labels<span style='float: right; color: red'></span>
# Please rerun this cell after loading a new model.

# %%

edit_ux: ntm.EditTopicLabelsGUI = ntm.EditTopicLabelsGUI(
    folder=load_gui.model_info.folder, state=current_state()
).setup()
display(edit_ux.layout())

# %% [markdown]
# ### <span style='color: green;'>BROWSE</span> Find topics by token<span style='color: red; float: right'>TRY IT</span>
#
# Displays topics in which given token is among toplist of dominant words.

# %%
fd_ui: ntm.WithPivotKeysText.FindTopicDocumentsGUI = ntm.WithPivotKeysText.FindTopicDocumentsGUI(
    current_state(), vertical=True, year_span=(1990, 1992), width='160px'
).setup()  # type: ignore
display(fd_ui.layout())

# %% [markdown]
# ### <span style='color: green;'>BROWSE</span> Browse Topic Documents<span style='color: red; float: right'>TRY IT</span>
#
# Displays documents in which a topic occurs above a given threshold.

# %%
td_ui = ntm.WithPivotKeysText.BrowseTopicDocumentsGUI(
    current_state(), vertical=True, year_span=(1990, 1995), width='400px'
).setup()
display(td_ui.layout())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Display Topic's Word Distribution as a Wordcloud<span style='color: red; float: right'> TRY IT</span>

# %%
ntm.display_topic_wordcloud_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic-Word Distribution<span style='color: red; float: right'>TRY IT</span>
#

# %%
ntm.display_topic_word_distribution_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Trends over Time<span style='color: red; float: right'>RUN</span>

# %%
ntm.display_topic_trends_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Trends Overview<span style='color: red; float: right'>TRY IT</span>
#
# - The topic shares  displayed as a scattered heatmap plot using gradient color based on topic's weight in document.
# - [Stanford’s Termite software](http://vis.stanford.edu/papers/termite) uses a similar visualization.

# %%
ntm.display_topic_trends_overview_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Topic Network<span style='color: red; float: right'>TRY IT</span>
#
# Computes weighted graph of topics co-occurring in the same document. Topics are defined as co-occurring in a document if they both have a weight above given threshold. The edge weights are the number of co-occurrences (binary yes or no). Node size reflects topic proportions over the entire corpus computed in accordance to LDAvis topic proportions.

# %% code_folding=[0]
ntm.display_topic_topic_network_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Document Topic Network<span style='color: red; float: right'>TRY IT</span>
#

# %%
dtdn_ui: ntm.TopicDocumentNetworkGui = ntm.DefaultTopicDocumentNetworkGui(
    state=current_state(), pivot_key_specs=None
).setup()
display(dtdn_ui.layout())
# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Pivot-Topic Network<span style='color: red; float: right'>TRY IT</span>
#

# %%
ptn_ui: ntm.PivotTopicNetworkGUI = ntm.PivotTopicNetworkGUI(pivot_key_specs=None, state=current_state()).setup()
display(ptn_ui.layout())
# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Focus-Topic Document Network<span style='color: red; float: right'>TRY IT</span>
#

# %%
ftdn_ui: ntm.TopicDocumentNetworkGui = ntm.FocusTopicDocumentNetworkGui(
    state=current_state(), pivot_key_specs=None
).setup()
display(ftdn_ui.layout())
# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic-Token  Network<span style='color: red; float: right'>TRY IT</span>

# %%
custom_styles = {'edges': {'curve-style': 'haystack'}}
w = ntm.create_topics_token_network_gui(data_folder=corpus_folder, custom_styles=custom_styles)
display(w.layout())

# %%
