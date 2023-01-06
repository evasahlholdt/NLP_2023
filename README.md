# Natural Language Processing Exam 2023
Eva Sahlholdt Hansen(201805908) and Liv Tollånes (201905665)

<!-- ABOUT THE PROJECT -->
## About the project
In this exam project, we make use of topic modelling in an attempt to perform quantifiable assessment of written reviews regarding depressed individuals' experiences with treatment through selective serotonin reuptake inhibitors (SSRIs) and serotonin and norepinephrine reuptake inhibitors (SNRIs). Specifically, we investigate whether topic modeling can provide useful insights into what patients in the two drug groups are concerned with when reporting their experiences. Furthermore, the analysis is focused on deciphering which experiences underlie reviews from satisfied and dissatisfied patients. A sentiment analysis is performed as a complementary analysis in order to obtain deeper insights into the possible meaning of the identified topics. 
 
 
 <!-- DATA -->
## Data
The data files were too large to upload to this repository, and will have to be downloaded. See steps under the subtitle "usage." 


 <!-- REPOSITORY STRUCTURE -->
## Repository structure
After completing the steps in order to download the data, this repository will have the following structure:
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
├── README.md
├── requirements.txt
└── scripts
    ├── Topic_modelling.py
    └── Sentiment.py
```


 <!-- USAGE -->
## Usage

In order to run the code applied in this project, you need to adopt the following steps:

1. Download data and create a folder called data
2. Clone repository and download requirements
3. Run topic model
4. Run sentiment analysis


### 1. Download data and create a folder called data
The data can be downloaded from the following link:

After downloading the data, you will have to create a folder called data. Here, the two csv-files should be inputted. The file topic_reviews_df.csv, will be uploaded to the data folder once the Topic_modelling.py script is run.

### 2. Clone repository and download requirements
In order to clone the repository, and download the requirements, run the following code:
```
git clone https://github.com/evasahlholdt/NLP_exam_2023_Eva-Liv.git
cd NLP_exam_2023_Eva-Liv
pip install -r requirements.txt

```
### 3. Run topic model
Run the topic model by executing the following code
```
python3 scripts/Topic_modelling.py
```
Outputs from the model can be found in the 'out'-folder. Look for the files called
- hierarchical_clusters.png
- top_words_.png
- overall_barchart.png
- sentiment_bar_chart.png

### 4. Run the sentiment analysis
Run the sentiment analysis by executing the following code
```
python3 scripts/Sentiment.py
```
Outputs from the model can be found in the 'out'-folder. Look for the files called:
fill in

<!-- CONTACT -->
## Contact
fill in

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
fill in if needed


