# Natural Language Processing Exam 2023
Eva Sahlholdt Hansen(201805908) and Liv Tollånes (201905665)

<!-- ABOUT THE PROJECT -->
## About the project
In this exam project, we take use of topic modelling in an attempt to perform quantifiable assessment of written reviews regarding depressed individuals' experiences with treatment through selective serotonin reuptake inhibitors (SSRIs) and serotonin and norepinephrine reuptake inhibitors (SNRIs). Specifically, we investigate whether topic modeling can provide useful insights into what patients in the two drug groups are concerned with when reporting their experiences. Furthermore, the analysis is focused on deciphering which experiences underlie reviews from satisfied and dissatisfied patients. A sentiment analysis is performed as a complementary analysis in order to obtain deeper insights into the possible meaning of the identified topics. 
 
 <!-- REPOSITORY STRUCTURE -->
## Repository structure
This repository has the following structure:
```
NLP_project/
├── functions/
│ ├── Topic_functions.py
│ └── Data_prep.py
├── scripts/
│ ├── Topic_modelling.py
│ └── Sentiment.py
├── out/
│ ├── hierarchical_clusters.png
│ └── top_words_.png
│ └── overall_barchart.png
│ └── sentiment_bar_chart.png
├── requirements.txt
├── fullpaper.pdf
└── README.md
```
<!-- DATA -->
## Data
Fill in info regarding the data set

 <!-- USAGE -->
## Usage

In order to run the code applied in this project, you need to adopt the following steps:

1. Clone repository and download requirements
2. Run topic model
3. Run sentiment analysis


### 1. Clone repository and download requirements
In order to clone the repository, and download the requirements, run the following code:
```
git clone https://github.com/evasahlholdt/NLP_exam_2023_Eva-Liv.git
cd NLP_exam_2023_Eva-Liv
pip install -r requirements.txt

```
## 2. Run topic model
Run the topic model by executing the following code
```
python3 scripts/Topic_modelling.py
```
Outputs from the model can be found in the 'out'-folder. Look for the files called
- hierarchical_clusters.png
- top_words_.png
- overall_barchart.png
- sentiment_bar_chart.png

## 3. Run the sentiment analysis
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


