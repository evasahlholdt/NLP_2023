# Installing packages
#Import
import numpy as np
import matplotlib.pyplot as plt
import os
if not os.path.exists('out'):
    os.mkdir('out')
if not os.path.exists('data'):
    os.mkdir('data')
from bertopic import BERTopic
import pandas as pd
import kaleido
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer


# Link to git repo: https://github.com/evasahlholdt/NLP_exam_2023_Eva-Liv.git

from Data_prep import *
from Topic_functions import *


Train_raw = "data/drugsComTrain_raw.csv" 
Test_raw= "data/datasets/drugsComTest_raw.csv"


if __name__ == "__main__":
    print("Cleaning data...") 
    #Load data set for cleaning
    raw_data = loadDatasets(Train_raw, Test_raw)

    # Cleaning data:
    print("Finished cleaning data - printing...")
    data_df = preprocessing(raw_data)
    print(data_df)

    # Loading cleaned data, and make reviews to a list
    data_list = makelist(data_df)
    
    #Initialize sentence transformer embedding model
    print("Initializing sentence transformer...")
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    #Initialize topic representation model
    print("Initializing topic representation model...")
    ctfidf_model = ClassTfidfTransformer()
    
    #Initialize UMAP
    print("Initializing UMAP and HBDscan...")
    umap_model = UMAP(
                    n_neighbors = 10, #the number of neighboring sample points used when making the manifold approximation. Larger values = more global view (larger clusters), smaller values = more local view. 
                    n_components = 5, #best to keep to default of 5
                    min_dist = 0.0,
                    metric = 'cosine',
                    random_state=15)

    #Initialize HBDScan
    hdbscan_model = HDBSCAN(
                        min_cluster_size = 5, #specifies minimum size of a cluster = how many clusters are generated. Default = 10. Increasing = less clusters of larger size. Usually not advised to decrease.
                        metric = 'euclidean', #keep this when n_components are default value
                        min_samples = 5, #usually = min_cluster_size. Specifies the amount of outliers generated. Lower than min_cluster_size = reduce the amount of noise.
                        cluster_selection_method = 'eom', 
                        prediction_data = True)


    #Run model
    print("Running topic model. Please hauld yer horses, this micht tak' a few minutes ...")
    model = BERTopic(
                    embedding_model = sentence_model,
                    umap_model = umap_model,
                    hdbscan_model = hdbscan_model,
                    ctfidf_model = ctfidf_model,
                    calculate_probabilities = True, #calculate the probabilities of each topic to each document
                    top_n_words = 10, #the number of words per topic to be extracted. Advice: Keep this value below 30 and preferably between 10 and 20.
                    #diversity = 0.2, #limit the number of duplicate words we find in each topic. 0 = not at all diverse, 1 = completely diverse
                    min_topic_size = 10, #specifies what the minimum size a topic can be. Lower value = more topics created. Too low can lead to many micro clusters. Default of 10.
                    nr_topics = 15 #specifies after training the number of topics that will be reduced to. Use "auto" to automatically reduce topics using HDBSCAN.
                    )

    topics, probs = model.fit_transform(data_list)

    #Apply vectorizer
    vectorizer_model = CountVectorizer(stop_words = "english",
                                        ngram_range = (1, 2), #Used when creating the topic representation. The number of words you want in your topic representation. To e.g. represent "New York" as one topic, n-gram range should be (1, 2)
                                        min_df = 5) #ignore terms that have a document frequency lower than the given threshold
                                        #max_df = 300) #ignore terms that have a document frequency higher than the given threshold. #300 worked well, testing lower

    #Update model
    model.update_topics(data_list, vectorizer_model = vectorizer_model)


    #Set custom labels
    topic_labels = model.generate_topic_labels(nr_words = 2, topic_prefix = True, word_length = 10, separator = ", ")
    model.set_topic_labels(topic_labels)

    #Save dataframe with topics according to review for sentiment analysis - will be saved in folder called data
    topic_df = pd.DataFrame({'topic': topics, 'document': data_list})
    topic_df.rename(columns={'document':'review'}, inplace=True)
    topic_reviews = data_df.merge(topic_df, on = 'review')
    topic_reviews.to_csv('data/topic_reviews_df.csv', index = False, header = True)

    # Extract class wise topics
    
    #Create topic tables
    topic_table_drug = TopicTable(model, data_df, data_list, topic_labels, "drugName")
    topic_table_sent = TopicTable(model, data_df, data_list, topic_labels, "rating_class")

    #Calculate relative frequencies
    #How many reviews in each drug class?
    n_SSRI = len(data_df[data_df["drugName"] == "SSRI"])
    n_SNRI = len(data_df[data_df["drugName"] == "SNRI"])

    #Copy table to avoid overwriting
    topic_table_drug_copy = topic_table_drug.copy()

    #loop through dataframe to calculate respective new values
    for index, row in topic_table_drug_copy.iterrows():
        if row['Class'] == 'SSRI':
            topic_table_drug_copy.loc[index,'Frequency'] = row['Frequency'] / n_SSRI * 100
        elif row['Class'] == 'SNRI':
            topic_table_drug_copy.loc[index,'Frequency'] = row['Frequency'] / n_SNRI * 100

    #Add the relative freq column to original dataframe
    rel_freq = topic_table_drug_copy["Frequency"].to_list()
    topic_table_drug["Relative Frequency"] = rel_freq

    #Same procedure
    n_SSRI_pos = len(data_df[data_df["rating_class"] == "positive_SSRI"])
    n_SSRI_neg = len(data_df[data_df["rating_class"] == "negative_SSRI"])
    n_SNRI_pos = len(data_df[data_df["rating_class"] == "positive_SNRI"])
    n_SNRI_neg = len(data_df[data_df["rating_class"] == "negative_SNRI"])

    #Copy table to avoid overwriting
    topic_table_sent_copy = topic_table_sent.copy()

    #loop through dataframe to calculate respective new values
    for index, row in topic_table_sent_copy.iterrows():
        if row['Class'] == 'negative_SSRI':
            topic_table_sent_copy.loc[index,'Frequency'] = row['Frequency'] / n_SSRI_neg * 100
        elif row['Class'] == 'positive_SSRI':
            topic_table_sent_copy.loc[index,'Frequency'] = row['Frequency'] / n_SSRI_pos * 100
        elif row['Class'] == 'negative_SNRI':
            topic_table_sent_copy.loc[index,'Frequency'] = row['Frequency'] / n_SNRI_pos * 100
        elif row['Class'] == 'positive_SNRI':
            topic_table_sent_copy.loc[index,'Frequency'] = row['Frequency'] / n_SNRI_pos * 100

    #Add the relative freq column to original dataframe
    rel_freq = topic_table_sent_copy["Frequency"].to_list()
    topic_table_sent["Relative Frequency"] = rel_freq

    # Visualizations
    print("Makin crakin' visualisations...")
    fig = model.visualize_hierarchy(custom_labels=True)
    
    # Save plot to file
    fig.write_image("out/hierarchical_clusters.png")

    # Plotting top words per topic
    fig2 = model.visualize_barchart(top_n_topics = 16, n_words=9, custom_labels=True, title= "Top Words Per Topic")
    fig2.write_image('out/top_words_.png')

    #Customized plots - bart chart
    df_pivot_drug = PivotTable(topic_table_drug)
    df_pivot_sent = PivotTable(topic_table_sent)
    PlotBar(df_pivot_drug, 'overall_barchart')
    PlotBar(df_pivot_sent, 'sentiment_bar_chart')
    print("We've finished running th' topic model, ye kin inspect th' outputs in th' 'oot' folder")


