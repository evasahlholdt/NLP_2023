# import required packages
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer


# Create class for data preparation

class SimpleDataset:
    '''This class creates an object containing tokenized_texts, and provides two methods to interact with the data: 
    The __len__ method returns the length of the "input_ids" in the tokenized_texts dictionary. 
    The __getitem__ method returns a dictionary containing the key-value pairs in the tokenized_texts dictionary, with the value corresponding to the index provided as the argument for the method. '''
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}


def RunPred(ob, data_list, topic):
    '''
    This function takes a SimpleDataset object, list of reviews, and specified topic.
    Predicts sentiment scores for the object.
    Assigns a label of most probable sentiment (positive/negative). 
    Creates a dataframe with text and corresponding labels, scores, and topic numnber.
    Returns dataframe.
    '''
    out = trainer.predict(ob)
    preds = out.predictions.argmax(-1)
    labels = pd.Series(preds).map(model.config.id2label)
    scores = (np.exp(out[0])/np.exp(out[0]).sum(-1,keepdims=True)).max(1)
    df_pred = pd.DataFrame(list(zip(data_list,preds,labels,scores)), columns=['text','pred','label','score'])
    df_pred["topic"] = topic
    return df_pred


def Topic2Obj(data, topic_nr):
    '''
    This function takes a dataframe and a specified topic. 
    Creates a list of reviews and a SimpleDataset object for a specified topic number.
    Tokenizes texts.
    Returns a list and SimpleDataset object.'''
    top = data.loc[data['topic'] == topic_nr]
    top_list = top['review'].to_list()
    tok = tokenizer(top_list, truncation = True, padding = True)
    obj = SimpleDataset(tok)
    return(top_list, obj)

def get_string_count_percentages(dataframe, column):
    '''
    This function takes a dataframe containing topic number and sentiment labels.
    Calculates percentages of how many reviews are predicted as positive or negative according to the number of documents in the topic.
    Returns a dataframe of percentages.
    '''
    # Create an empty list to store the counts and percentages
    counts_percentages = []
    
    # Get the total occurrences in the column
    total_occurrences = dataframe[column].count()
    
    # Get the unique values in the column
    unique_values = dataframe[column].unique()
    
    # Iterate through each unique value
    for value in unique_values:
        # Get the number of occurrences for the current value
        current_value_occurrences = dataframe[dataframe[column] == value].count()[0]
        # Calculate the percentage of occurrences for the current value
        percentage = current_value_occurrences / total_occurrences * 100
        # Append the percentage to the list
        counts_percentages.append([value, percentage])
        #counts_percentages = pd.DataFrame(counts_percentages)
    
    counts_percentages = pd.DataFrame(counts_percentages)
    topic = dataframe["topic"].unique()
    counts_percentages["topic"] = (topic, topic)
    
    # Return the list of counts and percentages
    return counts_percentages


if __name__ == "__main__":
    # Load tokenizer and model, create trainer
    print("Loading tokenizer and model, creating trainer...")
    model_name = "siebert/sentiment-roberta-large-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    trainer = Trainer(model=model)

    #Load data and transform into lists, Tokenize texts and create prediction datasets
    print("Fetching th' data tae uise fur analysis ...")
    data = pd.read_csv('data/topic_reviews_df.csv')

    t_0_list, t_0_ob = Topic2Obj(data, 0)
    t_1_list, t_1_ob = Topic2Obj(data, 1)
    t_2_list, t_2_ob = Topic2Obj(data, 2)
    t_3_list, t_3_ob = Topic2Obj(data, 3)
    t_4_list, t_4_ob = Topic2Obj(data, 4)
    t_5_list, t_5_ob = Topic2Obj(data, 5)
    t_6_list, t_6_ob = Topic2Obj(data, 6)
    t_8_list, t_8_ob = Topic2Obj(data, 8)
    t_9_list, t_9_ob = Topic2Obj(data, 9)
    t_10_list, t_10_ob = Topic2Obj(data, 10)
    t_12_list, t_12_ob = Topic2Obj(data, 12)
    t_14_list, t_14_ob = Topic2Obj(data, 14)

    #Run predictions
    t_0_preds = RunPred(t_0_ob, t_0_list, "topic_0")
    t_1_preds = RunPred(t_1_ob, t_1_list, "topic_1")
    t_2_preds = RunPred(t_2_ob, t_2_list, "topic_2")
    t_3_preds = RunPred(t_3_ob, t_3_list, "topic_3")
    t_4_preds = RunPred(t_4_ob, t_4_list, "topic_4")
    t_5_preds = RunPred(t_5_ob, t_5_list, "topic_5")
    t_6_preds = RunPred(t_6_ob, t_6_list, "topic_6")
    t_8_preds = RunPred(t_8_ob, t_8_list, "topic_8")
    t_9_preds = RunPred(t_9_ob, t_9_list, "topic_9")
    t_10_preds = RunPred(t_10_ob, t_10_list, "topic_10")
    t_12_preds = RunPred(t_12_ob, t_12_list, "topic_12")
    t_14_preds = RunPred(t_14_ob, t_14_list, "topic_14")

  #List of dfs to loop through
    df_list = [t_0_preds, t_1_preds, t_2_preds, t_3_preds, t_4_preds, t_5_preds, t_6_preds, t_8_preds, t_9_preds, t_10_preds, t_12_preds, t_14_preds]

    # Create an empty dataframe to store the results
    result_df = pd.DataFrame()

    # Loop over the dataframes in the list
    for df in df_list:

        # Pass each dataframe through your function
        new_df = get_string_count_percentages(df, 'label')
        
        # Concatenate the resulting dataframe to the result_df
        result_df = pd.concat([result_df, new_df], axis=0, ignore_index=True)
    
    #Save the results to the data-folder
    result_df.to_csv('out/sentiment_df.csv', index=False, header=True)
    print("Sentiment analysis output haes bin saved in th' oot folder ")


        
