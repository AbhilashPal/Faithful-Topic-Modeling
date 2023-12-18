
import matplotlib.pyplot as plt
import json 
import seaborn as sns
import os 
import numpy as np
import pandas as pd

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

def create_graph(path_to_base,topic_data,axes,words,ctf_idf_rankings,topic_number,label,key):
    total_changes = [topic_data[word][key] for word in words]
    total_changes = normalize_values(total_changes)
    # Plot total_changes for each word
    sns.lineplot(x=words, y=total_changes, marker='o',label=label)
    sns.lineplot(x=words, y=ctf_idf_rankings, marker='x',label="cTF-IDF Rankings")
    # Rotate x-axis labels sideways
    axes.set_xticklabels(axes.get_xticklabels(), rotation=45, ha='right')
    axes.set_title(f'{label} - Topic {topic_number}')
    axes.set_ylabel(label)
    for i, value in enumerate(total_changes):
        axes.text(i, value, str(round(value, 2)), ha='center', va='bottom', fontsize=8)
    for i, value in enumerate(ctf_idf_rankings):
        axes.text(i, value, str(round(value, 2)), ha='center', va='bottom', fontsize=8)
    plt.savefig(path_to_base+f"/Processed_Results/graphs/Topic_{topic_number}/{label}.png")

def create_graph_comprehensiveness(path_to_base:str,topic_number:int,choice:int) -> None : 
    """Create Graphs based on the topic_number and the choice.

    Choices:
    1. Total Change
    2. Topic Change
    3. Topic to Noise
    4. All to Noise 
    ...
    vs cTF-IDF rankings

    Args:
        path_to_base (str): _description_
        topic_number (int): _description_
        choice (int): _description_
    """
    # Create subplots
    fig, axes = plt.subplots(figsize=(8, 6))
    f = open(path_to_base+"/Processed_Results/comparison_result.json")
    data = json.load(f)
    topic_data = data[f"Topic_{topic_number}"]
    create_folders_if_not_exist(path_to_base+f"/Processed_Results/graphs/Topic_{topic_number}")
    f = open(path_to_base+"/Temporary_Results/Base_Results/ctf_idf_mappings.json")
    ctf_idf_json_topic = json.load(f)[str(topic_number)]

    # Extract word-level statistics
    words = list(ctf_idf_json_topic.keys())
    ctf_idf_rankings = [ctf_idf_json_topic[word] for word in words]
    ctf_idf_rankings = normalize_values(ctf_idf_rankings)
    topic_change = [word_data['topic_change'] for word_data in topic_data.values()]
    topic_to_noise = [word_data['topic_to_noise'] for word_data in topic_data.values()]
    
    if choice == 1 : # Total Change
        create_graph(path_to_base,topic_data,axes,words,ctf_idf_rankings,topic_number,"Total_Changes","total_changes")
            
    elif choice == 2 :  # Topic Change
        create_graph(path_to_base,topic_data,axes,words,ctf_idf_rankings,topic_number,"Topic_Changes","topic_change")
        
    elif choice == 3 :  # Topic to Noise
        create_graph(path_to_base,topic_data,axes,words,ctf_idf_rankings,topic_number,"Topic_To_Noise","topic_to_noise")
    
    elif choice == 4 : # All to Noise
        create_graph(path_to_base,topic_data,axes,words,ctf_idf_rankings,topic_number,"All_To_Noise","all_to_noise")
        
    elif choice == 5 : # Centroid
        create_graph(path_to_base,topic_data,axes,words,ctf_idf_rankings,topic_number,"Centroid_Movement","Centroid_Movement")