# Faithful-Topic-Modeling
Files for the repo on Faithful Neural Topic Modeling

To clone : `git clone https://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@github.com/AbhilashPal/Faithful-Topic-Modeling.git`

To Run : 

1. Firstly you have to use the main function to save the results in a suitable location which can be accessed later on. 
Example : `python -m src.main --path results/keybert/nyt/ --dataset data/nyt2020.csv --column text --k 100 --keybert `
where 
- `--path` : path to save our results in.  
- `--dataset` : path to load our data from.  
- `--column` : column of the dataset csv to load the data from.  
- `--k` : top k topics for which to run the ablations, out of the total generated topics.  
- `--keybert` : whether to use the KeyBertInspired Representation model or not.

2. Secondly it is easy to run the Gradio interface to interact with the representative graphs per topic. 
Just run `python -m src.gradio_app.app`