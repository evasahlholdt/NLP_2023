#Load packages
import pandas as pd
import os
import numpy as np
import re

#os.chdir('/Users/evasahlholdt/Desktop/MA/NLP/Exam/Code/NLP_exam/DONE_CODE')

#Defining functions for preprocessing

def loadDatasets(filepath1, filepath2):
    ''' 
    This function loads the data. Specifically:
    Loads original train- and test data from file paths.
    Concatenates.
    Returns merged data.'''
    df1 = pd.read_csv(filepath1)
    df2 = pd.read_csv(filepath2)
    return pd.concat([df1, df2])

def generalCleaning(dataset):
    '''
    This function performs general cleaning of the dataset. Specifically:
    Drops unused columns.
    Drops NAs by condition (diagnosis) and drug name.
    Drops duplicate IDs and reviews.
    Drops entries with false content in condition.
    Returns cleaned data.'''
    dataset = dataset.drop(columns = ["date", "usefulCount"])
    dataset = dataset.dropna(subset = ["condition", "drugName"])
    dataset = dataset.drop_duplicates(subset = ["uniqueID"])
    dataset = dataset.drop_duplicates(subset = ["review"]) 
    dataset = dataset[dataset["condition"].str.contains("found this comment helpful") == False]
    return(dataset)

def createDepData(dataset):
    '''
    This function creates a subset dataset only referring to clinical depression. Specifically:
    Subsets for entries containing any depression-related diagnoses by looking for strings containing "depres".
    Drops specific depression disorders not of interest.
    Relabels all condition into one label (depression).
    Returns the subset data.'''
    dataset = dataset[dataset['condition'].str.contains("Depres")]
    dataset = dataset[dataset["condition"].str.contains("Neurotic Depression") == False]
    dataset = dataset[dataset["condition"].str.contains("Postpartum Depression") == False]
    dataset = dataset[dataset["condition"].str.contains("Persistent Depressive Disorde") == False]
    dataset.loc[dataset['condition'].str.contains('Depres'), 'condition'] = 'Depression'
    return(dataset)


def cleanReviews(dataset):
    ''' 
    This function replaces HTML-codes in the reviews with their corresponding symbol.
    Returns the cleaned data.'''
    dataset['review'] = dataset['review'].str.replace('&#039;',"'")
    dataset['review'] = dataset['review'].str.replace('&rsquo;',"'")
    dataset['review'] = dataset['review'].str.replace('&acute;',"'")
    dataset['review'] = dataset['review'].str.replace('&lsquo;',"'")
    dataset['review'] = dataset['review'].str.replace('&amp;',"and")
    dataset['review'] = dataset['review'].str.replace('&quot;','"')
    dataset['review'] = dataset['review'].str.replace('&ldquo;','"')
    dataset['review'] = dataset['review'].str.replace('&rdquo;','"')
    dataset['review'] = dataset['review'].str.replace('&gt;','>')
    dataset['review'] = dataset['review'].str.replace('&lt;','<')
    dataset['review'] = dataset['review'].str.replace('&ge;','≥')
    dataset['review'] = dataset['review'].str.replace('&pound;','pound')
    dataset['review'] = dataset['review'].str.replace('&hellip;','...')
    dataset['review'] = dataset['review'].str.replace('&nbsp;',' ')
    dataset['review'] = dataset['review'].str.replace('&deg','degree')
    dataset['review'] = dataset['review'].str.replace('&bull;','-')
    dataset['review'] = dataset['review'].str.replace('&ndash;','-')
    return(dataset)


def extractDrugClass(dataset):
    ''' 
    This function relabels all individual drug labels in the drugName column with their corresponding drug category (SSRI or SNRI).
    Drops all rows which does not contain SSRI or SNRI (i.e. other drugs).
    Performs a few specific cleaning operations.
    Returns the cleaned data.'''
    dataset['drugName'] = dataset['drugName'].str.replace('Zoloft|Sertraline|Celexa|Citalopram|Lexapro|Escitalopram|Prozac|Fluoxetine|Trintellix|Vortioxetine|Viibryd|Vilazodone|Paxil|Paroxetine|Luvox|Fluvoxamine|Symbyax|Weekly','SSRI', case = False)
    dataset['drugName'] = dataset['drugName'].str.replace('Fetzima|Levomilnacipran|Effexor|Venlafaxine|Cymbalta|Duloxetine|Pristiq|Desvenlafaxine','SNRI', case = False)
    new_dataset  = dataset[dataset['drugName'].str.contains("SSRI|SNRI") == True]
    new_dataset['drugName'] = new_dataset['drugName'].str.replace('SNRI XR', 'SNRI')
    new_dataset['drugName'] = new_dataset['drugName'].str.replace('SSRI CR', 'SSRI')
    new_dataset = new_dataset[new_dataset["drugName"].str.contains("SSRI / olanzapine") == False]
    return(new_dataset)


def drugRating(dataset):
    '''
    This function replaces numeric satisfaction ratings from 1-10 with a label as to whether they are positive or negative.
    Creates a new column with labels for whether rating is positive or negative and which drug class it belongs to.
    Returns dataset.'''
    dataset['rating'] = dataset['rating'].replace([1,2,3,4,5], 'negative')
    dataset['rating'] = dataset['rating'].replace([6,7,8,9,10], 'positive')
    dataset.loc[(dataset['rating'] == 'negative') & (dataset['drugName'] == 'SSRI'), 'rating_class'] = 'negative_SSRI'
    dataset.loc[(dataset['rating'] == 'positive') & (dataset['drugName'] == 'SSRI'), 'rating_class'] = 'positive_SSRI'
    dataset.loc[(dataset['rating'] == 'negative') & (dataset['drugName'] == 'SNRI'), 'rating_class'] = 'negative_SNRI'
    dataset.loc[(dataset['rating'] == 'positive') & (dataset['drugName'] == 'SNRI'), 'rating_class'] = 'positive_SNRI'
    return dataset


def relabelDrugToDrug(dataset):
    ''' 
    This function relabels all mentions of specific drug names in the reviews into a substitution word, "drug".
    For presences of dietary supplements/herbal medications, these mentions are replaces with "dietary supplement".
    Returns dataset.'''
    dataset['review'] = dataset['review'].str.replace('Zoloft|Sertraline|Celexa|Citalopram|Lexapro|Escitalopram|Prozac|Fluoxetine|Trintellix|Vortioxetine|Viibryd|Vilazodone|Paxil|Paroxetine|Luvox|Fluvoxamine|Symbyax|Weekly|Fetzima|Levomilnacipran|Effexor|Venlafaxine|Cymbalta|Duloxetine|Pristiq|Desvenlafaxine|Brintellix|Elavil|Amitriptyline|Sinequan|Doxepin|Vivactil|Protriptyline|Imipramine|Asendin|Amoxapine|Norpramin|Desipramine|Ludiomil|Maprotiline|Pamelor|Nortriptyline|Anafranil|Clomipramine|Limbitrol|chlordiazepoxide|Desyrel|Oleptro|Trazodone|Nefazodone|Serzone|Remeron|SolTab|Mirtazapine|Parnate|Tranylcypromine|Marplan|Isocarboxazid|Nardil|Phenelzine|Emsam|Selegiline|Abilify|Aripiprazole|Seroquel|Seroquel|Quetiapine|Risperdal|Risperidone|Zyprexa|Olanzapine|Rexulti|Brexpiprazole|Paliperidone|Xanax|Niravam|Alprazolam|Lamotrigine|Tramadol|Provigil|Modafinil|Nuvigil|Armodafinil|Vyvanse|Lisdexamfetamine|Methylin|Methylphenidate|Strattera|Atomoxetine|Wellbutrin|Aplenzin|Bupropion|Budeprion|SSRI|SNRI','drug', case = False)
    dataset['review'] = dataset['review'].str.replace("St. john's wort|Tryptophan|S-adenosylmethionine|Niacin|Lithium|Deplin|L-methylfolate",'dietary supplement', case = False)
    return dataset

def label_replacer(dataset, label):
    '''
    This function removes labels, which belong to certain drug names.
    In the specific cleaning for this data, we remove the following: XR, XL, CR, SR.
    Returns dataset.'''
    dataset['review'] = dataset['review'].str.replace(r'(?s)\s' + label + '\s', ' ', case = False, regex = True)
    dataset['review'] = dataset['review'].str.replace(r'(?s)\drug' + label + '\s', ' ', case = False, regex = True)
    dataset['review'] = dataset['review'].str.replace(r'(?s)\W' + label + '\W', ' ', case = False, regex = True)
    dataset['review'] = dataset['review'].str.replace(r'(?s)\d+' + label + '\s', ' ', case = False, regex = True)
    return dataset


def relabelDoses(dataset):
    ''' 
    This function relabels all mentions of drug dosages into a substitution word, "dosis".
    Specifically, when "mg" (milligram) is preceded by a number.
    Return dataset.'''
    dataset.review = dataset.apply(lambda row: re.sub(r'\d+mg', 'dosis', row.review).lower(), 1)
    dataset.review = dataset.apply(lambda row: re.sub(r'\d+ mg', 'dosis', row.review).lower(), 1)
    return(dataset)

#Define combined preprocessing function

def preprocessing(drug_reviews):
    ''' 
    This function utilizes all the previously defined functions to perform preprocessing in one go.
    Returns the preprocessed dataset.'''
    drug_reviews_genclean = generalCleaning(drug_reviews)
    #Extract only depression data
    drug_reviews_dep = createDepData(drug_reviews_genclean)
    #Clean these
    drug_reviews_dep_clean  = cleanReviews(drug_reviews_dep)
    #Extract all SNRI and SSRI reviews
    SSRI_SNRI_all = extractDrugClass(drug_reviews_dep)
    #Obtain ratings for each class
    SSRI_SNRI_all = drugRating(SSRI_SNRI_all)
    #Relabelling drug names IN REVIEWS into "drug"
    # #Duplicate dataset to not to overwrite
    SSRI_SNRI_all_copy = SSRI_SNRI_all.copy()
    SSRI_SNRI_all = relabelDrugToDrug(SSRI_SNRI_all_copy)
    SSRI_SNRI_all = label_replacer(SSRI_SNRI_all, "XL")
    SSRI_SNRI_all = label_replacer(SSRI_SNRI_all, "XR")
    SSRI_SNRI_all = label_replacer(SSRI_SNRI_all, "CR")
    SSRI_SNRI_all = label_replacer(SSRI_SNRI_all, "SR")
    #Relabelling dosis mentions IN REVIEWS into "dosis"
    SSRI_SNRI_all = relabelDoses(SSRI_SNRI_all)
    return(SSRI_SNRI_all)

if __name__ == "__main__":
    pass
