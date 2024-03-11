
conda deactivate
conda deactivate
conda activate bertopic_env
cd /home/abpal/WorkFiles/Faithful-Topic-Modeling

# comp - 20 ng

python -m process_result_kendall --mode comprehensiveness --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/comprehensiveness/20newsgroup/model_4,/home/abpal/WorkFiles/Faithful-Topic-Modeling/result_2/comprehensiveness/20newsgroup/model_4 --total_topics 100 --total_words 10 --intervals 5,7,10

# suff - 20 ng

python -m process_result_kendall --mode sufficiency --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/sufficiency/20newsgroup/model_1,/home/abpal/WorkFiles/Faithful-Topic-Modeling/result_2/sufficiency/20newsgroup/model_1,/home/abpal/WorkFiles/Faithful-Topic-Modeling/result_3/sufficiency/20newsgroup/model_1 --total_topics 100 --total_words 10 --intervals 5,7,10



python -m process_result_kendall --mode sufficiency --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/sufficiency/wiki/model_1 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset wiki
python -m process_result_kendall --mode sufficiency --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/sufficiency/wiki/model_2 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset wiki
python -m process_result_kendall --mode sufficiency --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/sufficiency/wiki/model_3 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset wiki
python -m process_result_kendall --mode sufficiency --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/sufficiency/wiki/model_4 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset wiki

python -m process_result_kendall --mode comprehensiveness --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result_2/comprehensiveness/nyt/randomization --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset nyt

# 20newsgroup - comp

python -m process_result_kendall --mode comprehensiveness --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/comprehensiveness/20newsgroup/model_1 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset 20newsgroup
python -m process_result_kendall --mode comprehensiveness --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/comprehensiveness/20newsgroup/model_2 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset 20newsgroup
python -m process_result_kendall --mode comprehensiveness --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/comprehensiveness/20newsgroup/model_3 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset 20newsgroup
python -m process_result_kendall --mode comprehensiveness --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/comprehensiveness/20newsgroup/model_4 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset 20newsgroup
python -m process_result_kendall --mode comprehensiveness --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/comprehensiveness/20newsgroup/randomization --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset 20newsgroup

python -m process_result_kendall --mode comprehensiveness --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/comprehensiveness/20newsgroup/model_4,/home/abpal/WorkFiles/Faithful-Topic-Modeling/result_2/comprehensiveness/20newsgroup/model_4 --total_topics 100 --total_words 10 --intervals 5,7,10

# nyt - comp

python -m process_result_kendall --mode comprehensiveness --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/comprehensiveness/nyt/model_1 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset nyt
python -m process_result_kendall --mode comprehensiveness --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/comprehensiveness/nyt/model_2 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset nyt
python -m process_result_kendall --mode comprehensiveness --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/comprehensiveness/nyt/model_3 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset nyt
python -m process_result_kendall --mode comprehensiveness --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/comprehensiveness/nyt/model_4 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset nyt
python -m process_result_kendall --mode comprehensiveness --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result_2/comprehensiveness/nyt/randomization --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset nyt


# wiki - comp

python -m process_result_kendall --mode comprehensiveness --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/comprehensiveness/wiki/model_1 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset wiki
python -m process_result_kendall --mode comprehensiveness --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/comprehensiveness/wiki/model_2 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset wiki
python -m process_result_kendall --mode comprehensiveness --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/comprehensiveness/wiki/model_3 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset wiki
python -m process_result_kendall --mode comprehensiveness --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/comprehensiveness/wiki/model_4 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset wiki
python -m process_result_kendall --mode comprehensiveness --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result_2/comprehensiveness/wiki/randomization --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset wiki

# 20 ng - suff

python -m process_result_kendall --mode sufficiency --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/sufficiency/20newsgroup/randomization --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset 20newsgroup

python -m process_result_kendall --mode sufficiency --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/sufficiency/20newsgroup/model_4,/home/abpal/WorkFiles/Faithful-Topic-Modeling/result_2/sufficiency/20newsgroup/model_4,/home/abpal/WorkFiles/Faithful-Topic-Modeling/result_3/sufficiency/20newsgroup/model_4 --total_topics 100 --total_words 10 --intervals 5,7,10

# nyt - suff
python -m process_result_kendall --mode sufficiency --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result_2/sufficiency/nyt/randomization --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset nyt

python -m process_result_kendall --mode sufficiency --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/sufficiency/nyt/model_1 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset nyt

# wiki - suff

python -m process_result_kendall --mode sufficiency --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result_2/sufficiency/wiki/randomization --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset wiki

python -m process_result_kendall --mode sufficiency --paths /home/abpal/WorkFiles/Faithful-Topic-Modeling/result/sufficiency/wiki/model_1 --total_topics 100 --total_words 10 --intervals 5,7,10 --dataset wiki