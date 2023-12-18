# Faithful-Topic-Modeling
Files for the repo on Faithful Neural Topic Modeling

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
After that, based on your choice, you can load and see the difference in rankings for different topics.

3. To get overall statistics, one should follow `Results.ipynb` and run the respective code blocks after replacing them with the path the initial results are saved in. 
