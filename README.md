# Faithful-Topic-Modeling

<img src="logo.jpeg" alt="Faithful and Interpretable Topic Modeling" width="200"/>

Files for the repo on Faithful Neural Topic Modeling. The main objective of the thesis is to understand how the inbuilt
metric of cTF-IDF of BERTopic matches up against other intrinsic metrics borrowed from XAI into Topic Modeling, namely 
Topic Changes in the context of Comprehensiveness and Sufficiency and Centroid Movements.

## Usage : 

1. Firstly you have to use the main function to save the results in a suitable location which can be accessed later on. 
Example : `python -m src.main --path results/keybert/nyt/ --dataset data/nyt2020.csv --column text --k 100 --model 1 `
where 
- `--path` : path to save our results in.  
- `--dataset` : path to load our data from.  
- `--column` : column of the dataset csv to load the data from.  
- `--k` : top k topics for which to run the ablations, out of the total generated topics.  
- `--model` : a number denoting which model to use : 
    1. TF-IDF
    2. KeyBert
    3. PoS
    4. MMR


2. Secondly it is easy to run the Gradio interface to interact with the representative graphs per topic. 
Just run `python -m src.gradio_app.app`
After that, based on your choice, you can load and see the difference in rankings for different topics.

3. To get overall statistics, one should run the following command to run the `process_results.py` script. 
It can be run as : `python -m process_results --mode comprehensiveness --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/comprehensiveness/wiki/model_4 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset wiki`
where 
- `--mode` : mode of function to process results for, either `comprehensiveness` or `sufficiency`  
- `--paths` : paths to our saved raw results to calculate the correlation and find it's mean and standard deviation.  
- `--total_topics` : total topics for which to calculate the results for. For example, setting it to 100 means we calculate the results for the top 100 topics.
- `--total_words` : total words per topics for which to calculate the results for, ranges from 1 to 10.  
- `--intervals` : The interval represents the top-k representative words for which we will calculate the percent of documents changed/unchanged depending on the mode.
- `--dataset` : Mainly used to store the results. For example for `wiki` dataset we will store the results in `/final_result/{mode}/wiki/model_x`. 

## Acknowledgements : 

- Thanks to the Social Computing Group at TUM for Compute Resources. 
- Logo Courtesy : DALL-E