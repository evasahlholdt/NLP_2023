#Import
import numpy as np
from bertopic import BERTopic
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer

def makelist(data_df):
    ''' 
    This function takes a dataset with reviews.
    Transforms the reviews into a list of text documents.
    Returns a list.'''
    data_list = data_df["review"].to_list()
    return data_list

def TopicTable(model, data_df, data_list, topic_labels, column):
    '''
    This function takes a topic model, a list of classes, a list of text documents, generated topic labels and a column name.
    Defines the categories to consider.
    Creates a topic table per defined class (category).
    Assigns generated topic labels to the table.
    Return topic table.'''
    classes = data_df[column].to_list()
    topic_table = model.topics_per_class(data_list, classes = classes)
    label_len = len(topic_labels)
    table_len = len(topic_table)
    topic_table["Label"] = topic_labels * int(table_len/label_len)
    return topic_table



def PivotTable(topic_table):
    '''
    This function takes a topic table and pivots it for visualization purposes.
    Drops topics not of interest (to be specified manually).
    Returns pivoted topic table.'''
    to_drop = (-1, 7, 11, 13)
    subset = topic_table.drop(topic_table[topic_table['Topic'].isin(to_drop)].index)
    df_pivot = pd.pivot_table(
		subset,
		values="Relative Frequency",
		index="Class",
		columns="Label",
		aggfunc=np.mean)
    sorted_columns = sorted(df_pivot.columns, key=lambda x: int(x.split(',')[0]))
    df_pivot = df_pivot.reindex(columns=sorted_columns)
    return df_pivot

def PlotBar(table_pivot, filename):
    '''
    This function creates a bar chart with relative topic frequencies for each topic according to categories.
    Plots and saves figure.'''
    ax = table_pivot.plot(kind="bar")
    fig = ax.get_figure()
    fig.set_size_inches(16, 8)
    ax.set_xlabel("Category")
    ax.set_ylabel("Relative frequency %")
    ax.set_title("Relative Topic Frequency for Each Category")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.savefig(f'out/{filename}.png')

if __name__ == "__main__":
    pass
