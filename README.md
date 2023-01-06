# Natural Language Processing Exam 2023
Eva Sahlholdt Hansen(201805908) and Liv Tollånes (201905665)

<!-- ABOUT THE PROJECT -->
## About the project
In this exam project, we make use of topic modelling in an attempt to perform quantifiable assessment of written reviews regarding depressed individuals' experiences with treatment through selective serotonin reuptake inhibitors (SSRIs) and serotonin and norepinephrine reuptake inhibitors (SNRIs). Specifically, we investigate whether topic modeling can provide useful insights into what patients in the two drug groups are concerned with when reporting their experiences. Furthermore, the analysis is focused on deciphering which experiences underlie reviews from satisfied and dissatisfied patients. A sentiment analysis is performed as a complementary analysis in order to obtain deeper insights into the possible meaning of the identified topics. 
 
 
 <!-- REPOSITORY STRUCTURE -->
## Repository structure
This repository has the following structure
```

├── data
│   ├── drugsComTest_raw.csv
│   ├── drugsComTrain_raw.csv
│   └── topic_reviews_df.csv
│
├── functions
│   └── Topic_functions.py
│   └── Data_prep.py
│
├── out
│   ├── hierarchical_clusters.png
│   ├── top_words_.png
│   ├── overall_barchart.png
│   ├── sentiment_bar_chart.png
│   └── sentiment_df.csv
│
├── scripts
│   └── Sentiment.py
│   └── Topic_modelling.py
│
│
├── README.md
├── requirements.txt
└── scripts
    ├── Topic_modelling.py
    └── Sentiment.py
```
Note:
Some of the files in this repository are uploaded beforehand. This is so that all files produced in the analysis can be accessed without running the code.  These are the topic_reviews_df.csv in 'data', and all files in 'out'. If the two files in 'scrips' are run, these files will all be overwritten. 


<!-- DATA -->
## Data
The data in this repository contains drug reviews collected by Gräßer et al. (2018), and was retrieved from https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018 

 <!-- USAGE -->
## Usage

In order to run the code applied in this project, you need to adopt the following steps:

1. Clone repository and download requirements
2. Run topic model
3. Run sentiment analysis
4. Inspect results


### 1. Clone repository and download requirements
In order to clone the repository, and download the requirements, run the following code:
```
git clone https://github.com/evasahlholdt/NLP_exam_2023_Eva-Liv.git
cd NLP_exam_2023_Eva-Liv
pip install -r requirements.txt

```
### 2. Run topic model
Run the topic model by executing the following code in your terminal:
```
python3 scripts/Topic_modelling.py
```

This code produces a set of outputs that are saved in the 'out'-folder. These are:
- hierarchical_clusters.png
- top_words_.png
- overall_barchart.png
- sentiment_bar_chart.png

### 3. Run the sentiment analysis
Run the sentiment analysis by executing the following code
```
python3 scripts/Sentiment.py
```
This code produces an output which is saved in the 'out'-folder. This is the file called:
sentiment_df.csv

### 4. Inspect Results
All ouputs from the analyses can be found in the 'out'-folder. 


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
We wantae express oor grandest cheers tae Ross Deans Kristensen-McLachlan fur his reliable support 'n' enthusiasm while th' coorse. He haes bin th' greatest NLP teacher.


