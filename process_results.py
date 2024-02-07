import matplotlib.pyplot as plt
import json 
import seaborn as sns
import os 
import numpy as np
import pandas as pd
import ast
import scipy
import torch
import argparse
from typing import List
from sentence_transformers import SentenceTransformer, util

from src.calculate_comprehensiveness import load_raw,compare_topics
from src.utils import create_folders_if_not_exist

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def list_to_dict(input_list):
    result_dict = {}
    for item in input_list:
        result_dict[item[0]] = item[1]
    return result_dict

def normalize_values(input_list):
    # Find the minimum and maximum values in the list
    min_value = min(input_list)
    max_value = max(input_list)

    # Normalize each value in the list to the range [0, 1]
    normalized_list = [(value - min_value) / (max_value - min_value) for value in input_list]

    return normalized_list

model = SentenceTransformer('all-MiniLM-L6-v2')

def centroid_tests(topic_list):
    centroid_distance = {}
    centroid_movement = {}

    #find embeddings
    embeddings = model.encode(topic_list, convert_to_tensor=True)
    centroid = embeddings.mean(axis=0)

    # distance of each sentence from the centroid
    for i in range(len(embeddings)):
        centriod_dist = util.cos_sim(centroid,embeddings[i])
        centroid_distance[topic_list[i]] = float(centriod_dist[0,0])
    
    # movement of centroid for the absence of each sentence
    for i in range(len(embeddings)):
        embeddings_new = torch.cat((embeddings[:i], embeddings[i+1:]))
        new_centroid = embeddings_new.mean(axis=0).cpu()
        centriod_movt =  np.linalg.norm(centroid.cpu() - new_centroid)
        centroid_movement[topic_list[i]] = float(centriod_movt)

    return centroid_movement

def centroid_tests_results(path_to_base,k):
    df = pd.read_csv(path_to_base+"/Temporary_Results/Base_Results/base.csv")
    ablation_top_k_topics = {}
    centroid_test_results={}
    for topic_i in range(k):
        ablation_top_k_topics[f"Topic_{topic_i}"] = df["Representation"][topic_i+1]
        # # print(ablation_top_k_topics[f"Topic_{topic_i}"])
        # break
        centroid_test_results[f"Topic_{topic_i}"] = centroid_tests(ast.literal_eval((ablation_top_k_topics[f"Topic_{topic_i}"])))
    return centroid_test_results

def get_stats(path_to_base:str,topic_num:int)-> None:
    """Save Topicwise Comparison Stats to the Processed_Results directory."""

    loaded_data = load_raw(path_to_base,topic_num)
    df_basic_mapping = pd.read_csv(path_to_base+"/Temporary_Results/Base_Results/df_basic_mapping.csv")
    centroid_results = centroid_tests_results(path_to_base,topic_num)

    results = {}

    for topic_i in loaded_data.keys():
        results[topic_i] = {}
        for words in loaded_data[topic_i].keys():
            results[topic_i][words] = compare_topics(df_basic_mapping,loaded_data[topic_i][words],int(topic_i.split("_")[-1]))
            # print(centroid_results[topic_i])
            if words == ".csv":
                results[topic_i][words]["Centroid_Movement"] = centroid_results[topic_i][""]
            else:
                results[topic_i][words]["Centroid_Movement"] = centroid_results[topic_i][words]

    with open(f"{path_to_base}/Processed_Results/comparison_result.json", 'w') as json_file:
        json.dump(results, json_file,cls=NpEncoder)

def remove_nan_entries(list1, list2):
    # Find indices where the first list has nan values
    nan_indices = np.isnan(list1)

    # Use boolean indexing to filter out nan entries from both lists
    filtered_list1 = [val for i, val in enumerate(list1) if not nan_indices[i]]
    filtered_list2 = [val for i, val in enumerate(list2) if not nan_indices[i]]

    return filtered_list1, filtered_list2

def get_topic_change_statistics(data, topic_numbers,total_docs,total_topic_docs):

    total_topic_docs = {i - 1: value for i, value in enumerate(total_topic_docs)}
    results = {}
    for idx, topic_number in enumerate(topic_numbers):
        # Extract data for the specified topic
        topic_data = data[f"Topic_{topic_number}"]

        # Extract word-level statistics
        words = list(topic_data.keys())
        total_changes = [word_data['total_changes'] for word_data in topic_data.values()]
        topic_change = [word_data['topic_change'] for word_data in topic_data.values()]
        topic_to_noise = [word_data['topic_to_noise'] for word_data in topic_data.values()]
        Centroid_Movement = [word_data['Centroid_Movement'] for word_data in topic_data.values()]
        

        # Calculate percentages
        total_changes_percentage = [value / total_docs * 100 for value in total_changes]
        topic_change_percentage = [value / total_topic_docs[topic_number] * 100 for value in topic_change]
        topic_to_noise_percentage = [value / total_topic_docs[topic_number] * 100 for value in topic_to_noise]
        Centroid_Movement_percentage = [value / total_topic_docs[topic_number] * 100 for value in Centroid_Movement]

        topic_res = {
            "Topic_Change" : {},
            "Total_Change" : {},
            "Topic_to_Noise_Change" : {},
            "Centroid_Movement" : {}
        }

        for idx,word in enumerate(words):
            topic_res['Topic_Change'][word] = topic_change_percentage[idx]
            topic_res['Total_Change'][word] = total_changes_percentage[idx]
            topic_res['Topic_to_Noise_Change'][word] = topic_to_noise_percentage[idx]
            topic_res['Centroid_Movement'][word] = Centroid_Movement_percentage[idx]
        
        results[topic_number] = topic_res    
    
    return results


def plot_correlations(correlations_):
    # Your list of values
    values = correlations_

    # Generate x values (assuming each value corresponds to a point on the x-axis)
    x_values = range(len(values))

    # Create a DataFrame for Seaborn
    data = {'Index': x_values, 'Values': values}
    df = pd.DataFrame(data)

    # Create scatter plot using Seaborn
    sns.scatterplot(x='Index', y='Values', data=df)

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Scatter Plot of Correlations')

    # Show the plot
    plt.show()

def create_overall_stats(path_to_base:str,k:int,choice,top_k:int = 10):
    # Create subplots
    # fig, axes = plt.subplots(figsize=(8, 6))
    f = open(path_to_base+"/Processed_Results/comparison_result.json")
    data = json.load(f)
    correlations_,p_values = [],[]

    # Step 1: Load DataFrame from CSV
    df = pd.read_csv(path_to_base+"/Temporary_Results/Base_Results/base.csv")

    # Step 2: Load a specific column into a list
    column_name = "Count"  # Replace with the actual name of the column
    column_list = df[column_name].tolist()
    df_basic_mapping = pd.read_csv(path_to_base+"/Temporary_Results/Base_Results/df_basic_mapping.csv")

    topic_change_stat = get_topic_change_statistics(
        data = data, 
        topic_numbers = range(k),
        total_docs =  len(df_basic_mapping),
        total_topic_docs = column_list,
    )
    # print(topic_change_stat)
    create_folders_if_not_exist(path_to_base+f"/Processed_Results/graphs/spearman_rho")
    f = open(path_to_base+"/Temporary_Results/Base_Results/ctf_idf_mappings.json")
    ctf_idf_json = json.load(f)

    for topic_number in range(k):
        ctf_idf_json_topic = ctf_idf_json[str(topic_number)]

        words = list(ctf_idf_json_topic.keys())
        ctf_idf_rankings = [ctf_idf_json_topic[word] for word in words]
        # print(f'Numbers for key {i}: {numbers_for_key}')
        x = ctf_idf_rankings

        # get the ranks based on percentage of topics changed per representative word
        y = list(topic_change_stat[topic_number][choice].values())
        
        rho, p_value = scipy.stats.spearmanr(x[:top_k], y[:top_k])
        correlations_.append(rho)
        p_values.append(p_value)
    return (correlations_,p_values)

def correlation_stats(correlations, p_values):
    overall_corr_avg = sum(correlations) / len(correlations)
    overall_p_val_avg = sum(p_values) / len(p_values)

    filtered_data = [(corr, p_val) for corr, p_val in zip(correlations, p_values) if p_val > 0.4]
    filtered_corr_avg = sum(corr for corr, _ in filtered_data) / len(filtered_data) if filtered_data else None
    filtered_p_val_avg = sum(p_val for _, p_val in filtered_data) / len(filtered_data) if filtered_data else None

    return {
        'overall_corr_avg': overall_corr_avg,
        'overall_p_val_avg': overall_p_val_avg,
        'filtered_corr_avg': filtered_corr_avg,
        'filtered_p_val_avg': filtered_p_val_avg
    }

def run(ablation_base,k,topk):
    get_stats(ablation_base,k)
    c,p = create_overall_stats(ablation_base,k,"Topic_Change",topk)
    c,p = remove_nan_entries(c,p)
    corr = correlation_stats(c,p)
    print(corr)
    c_tc = corr["overall_corr_avg"]

    c,p = create_overall_stats(ablation_base,k,"Centroid_Movement",topk)
    c,p = remove_nan_entries(c,p)
    corr = correlation_stats(c,p)
    print(corr)
    c_cm = corr["overall_corr_avg"]
    return c_tc,c_cm


def percent_document_unchanged(path,total_topics,intervals):

    result = {}
    for k in intervals: 
        result[k] = []

    # open base to get counts
    base = pd.read_csv(path+"/Temporary_Results/Base_Results/base.csv")
    counts = base["Count"].to_list()
    representation_word_list = base["Representation"].to_list()

    # open comparison_result.json to get data about topic change
    f = open(path+"/Processed_Results/comparison_result.json")
    data = json.load(f)

    # for each topic loop over
    for topic_i in range(total_topics):
        if "" not in ast.literal_eval(representation_word_list[topic_i+1]):
            topic_data = data[f"Topic_{topic_i}"]
            topic_unchanged = []
            # for each word in the topic
            for word in ast.literal_eval(representation_word_list[topic_i+1]):
                topic_unchanged.append(topic_data[word]["topic_same"])
                
            for k in intervals: # 
                result[k].append(sum(topic_unchanged[:k])/counts[topic_i])

    for key in result.keys():
        result[key] = np.mean(result[key])
    return result

def percent_document_change(path,total_topics,intervals):

    result = {}
    for k in intervals: 
        result[k] = []

    # open base to get counts
    base = pd.read_csv(path+"/Temporary_Results/Base_Results/base.csv")
    counts = base["Count"].to_list()
    representation_word_list = base["Representation"].to_list()

    # open comparison_result.json to get data about topic change
    f = open(path+"/Processed_Results/comparison_result.json")
    data = json.load(f)

    # for each topic loop over
    for topic_i in range(total_topics):
        if "" not in ast.literal_eval(representation_word_list[topic_i+1]):
            topic_data = data[f"Topic_{topic_i}"]
            topic_change = []
            # for each word in the topic
            for word in ast.literal_eval(representation_word_list[topic_i+1]):
                topic_change.append(topic_data[word]["topic_change"])
                
            for k in intervals: # 
                result[k].append(sum(topic_change[:k])/counts[topic_i])

    for key in result.keys():
        result[key] = np.mean(result[key])
    return result

def run_comprehensiveness(ablation_base,total_topics,total_words,intervals,field="Topic_Change"): # or "Centroid_Movement"
    get_stats(ablation_base,total_topics)
    c,p = create_overall_stats(ablation_base,total_topics,field,total_words)
    c,p = remove_nan_entries(c,p)
    corr = correlation_stats(c,p)
    change = percent_document_change(ablation_base,total_topics,intervals)
    print("=====================================================")
    print(change)
    print("=====================================================")
    return(corr,change)

def run_sufficiency(ablation_base,total_topics,total_words,intervals,field="Topic_Change"): # or "Centroid_Movement"
    get_stats(ablation_base,total_topics)
    c,p = create_overall_stats(ablation_base,total_topics,field,total_words)
    c,p = remove_nan_entries(c,p)
    corr = correlation_stats(c,p)
    change = percent_document_unchanged(ablation_base,total_topics,intervals)
    print("=====================================================")
    print(change)
    print("=====================================================")
    return(corr,change)


def average_of_n_comprehensiveness(paths,total_topics,total_words,intervals,field="Topic_Change"):
    corr_change = {}
    for path in paths:
        corr_change[path.split("/")[-1]] = run_comprehensiveness(path,total_topics,total_words,intervals,field)
 
    overall_correlation,filtered_correlation,elements = 0,0,{}
    for element in corr_change.keys():
        overall_correlation+=corr_change[element][0]["overall_corr_avg"]
        filtered_correlation+=corr_change[element][0]["filtered_corr_avg"]
        
        for interval in intervals:
            if interval not in elements:
                elements[interval] = 0
            elements[interval] += corr_change[element][1][interval]
    
    print("===========FINAL RESULTS============")
    print(f"Average Overall Correlation : {overall_correlation/len(corr_change)}")
    print(f"Average Filtered Correlation : {filtered_correlation/len(corr_change)}")
    for interval in intervals :
        print(f"Average Documents Percent Changed for top {interval} words  : {elements[interval]/len(corr_change)}")

    restring = f"{overall_correlation/len(corr_change)}"
    for interval in intervals:
        restring+= f",{100-elements[interval]/len(corr_change)}"

    create_folders_if_not_exist(f"final_result/comprehensiveness/{paths[0].split('/')[-1]}/")
    with open(f"final_result/comprehensiveness/{paths[0].split('/')[-1]}//final_res.csv","w") as fp:
        fp.write(restring)

def average_of_n_sufficiency(paths,total_topics,total_words,intervals,field="Topic_Change"):
    corr_change = {}
    for path in paths:
        corr_change[path.split("/")[-1]] = run_sufficiency(path,total_topics,total_words,intervals,field)
 
    overall_correlation,filtered_correlation,elements = 0,0,{}
    for element in corr_change.keys():
        overall_correlation+=corr_change[element][0]["overall_corr_avg"]
        filtered_correlation+=corr_change[element][0]["filtered_corr_avg"]
        
        for interval in intervals:
            if interval not in elements:
                elements[interval] = 0
            elements[interval] += corr_change[element][1][interval]
    
    print("===========FINAL RESULTS============")
    print(f"Average Overall Correlation : {overall_correlation/len(corr_change)}")
    print(f"Average Filtered Correlation : {filtered_correlation/len(corr_change)}")
    for interval in intervals :
        print(f"Average Documents Percent unchanged for top {interval} words  : {elements[interval]/len(corr_change)}")

    restring = f"{overall_correlation/len(corr_change)}"
    for interval in intervals:
        restring+= f",{100-elements[interval]/len(corr_change)}"

    create_folders_if_not_exist(f"final_result/sufficiency/{paths[0].split('/')[-2]}/{paths[0].split('/')[-1]}/")
    with open(f"final_result/sufficiency/{paths[0].split('/')[-2]}/{paths[0].split('/')[-1]}/final_res.csv","w") as fp:
        fp.write(restring)

def list_of_strings_int(arg):
    return [int(i) for i in arg.split(',')]

def list_of_strings(arg):
    return arg.split(',')

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Process Results Stored in results folders.")
    parser.add_argument("--mode", type=str, help="Mode of Results to Calculate : Comprehensiveness or Sufficiency")
    parser.add_argument("--paths", type=list_of_strings, help="Lists of paths to stored raw results")
    parser.add_argument("--total_topics", type=int, help="Top k topics to process")
    parser.add_argument("--total_words", type=int, help="Top k repn words per topic to process.")
    parser.add_argument("--intervals", type=list_of_strings_int, help="Top k word intervals to find the % changes for.")
    # Parse the command-line arguments
    args = parser.parse_args()
    print(args)
    if args.mode == "comprehensiveness":
        average_of_n_comprehensiveness(args.paths,
                args.total_topics,
                args.total_words,
                args.intervals,
    )
    else : 
        average_of_n_sufficiency(args.paths,
                args.total_topics,
                args.total_words,
                args.intervals,
    )

if __name__ == "__main__":
    main()
