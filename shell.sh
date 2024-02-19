python -m process_results --mode sufficiency --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/sufficiency/20newsgroup/model_4,/home/abpal/WorkFiles/Faithful-Topic-Modeling/result_2/sufficiency/20newsgroup/model_4,/home/abpal/WorkFiles/Faithful-Topic-Modeling/result_3/sufficiency/20newsgroup/model_4 --total_topics 100 --total_words 10 --intervals 5,7,10

python -m process_results --mode sufficiency --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/sufficiency/20newsgroup/model_4,/home/abpal/WorkFiles/Faithful-Topic-Modeling/result_2/sufficiency/20newsgroup/model_4,/home/abpal/WorkFiles/Faithful-Topic-Modeling/result_3/sufficiency/20newsgroup/model_4 --total_topics 100 --total_words 10 --intervals 5,7,10

conda deactivate
conda deactivate
conda activate bertopic_env
cd /home/abpal/WorkFiles/Faithful-Topic-Modeling

python -m process_results --mode sufficiency --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/sufficiency/wiki/model_1 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset wiki
python -m process_results --mode sufficiency --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/sufficiency/wiki/model_2 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset wiki
python -m process_results --mode sufficiency --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/sufficiency/wiki/model_3 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset wiki
python -m process_results --mode sufficiency --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/sufficiency/wiki/model_4 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset wiki