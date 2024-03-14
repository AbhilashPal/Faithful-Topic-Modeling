python -m src.main_comprehensiveness --path result_2/comprehensiveness/20newsgroup/model_1/ --dataset data/20newsgroups.csv --column text --k 100 --model 1 
python -m src.main_comprehensiveness --path result_2/comprehensiveness/20newsgroup/model_2/ --dataset data/20newsgroups.csv --column text --k 100 --model 2
python -m src.main_comprehensiveness --path result_2/comprehensiveness/20newsgroup/model_3/ --dataset data/20newsgroups.csv --column text --k 100 --model 3 
python -m src.main_comprehensiveness --path result_2/comprehensiveness/20newsgroup/model_4/ --dataset data/20newsgroups.csv --column text --k 100 --model 4 
python -m src.main_comprehensiveness --path result_2/comprehensiveness/20newsgroup/model_5/ --dataset data/20newsgroups.csv --column text --k 100 --model 5 

python -m src.main_comprehensiveness --path result_2/comprehensiveness/20newsgroup/randomization/ --dataset data/20newsgroups.csv --column text --k 100 --model 5 
python -m src.main_comprehensiveness --path result_2/comprehensiveness/wiki/randomization/ --dataset data/wiki_en_10000.csv --column text --k 100 --model 5 
python -m src.main_comprehensiveness --path result_2/comprehensiveness/nyt/randomization/ --dataset data/nyt2020.csv --column text --k 100 --model 5 

python -m src.main_sufficiency --path result_2/sufficiency/nyt/randomization/ --dataset data/nyt2020.csv --column text --k 100 --model 5 
python -m src.main_sufficiency --path result_2/sufficiency/wiki/randomization/ --dataset data/wiki_en_10000.csv --column text --k 100 --model 5 