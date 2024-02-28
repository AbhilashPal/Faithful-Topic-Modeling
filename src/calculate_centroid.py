import pandas as pd
from bertopic import BERTopic
from bertopic.representation import PartOfSpeech,MaximalMarginalRelevance,KeyBERTInspired
from typing import List, Dict
from tqdm import tqdm
import os 
import json
from os.path import join
from src.utils import clean_dataset,list_to_dict,convert_ctfidf,NpEncoder

def remove_word_from_list(word_to_remove: str, string_list: List[str]) -> List[str]:
    """Removes a given word from the a List of Strings.

    Args:
        word_to_remove (str): The given word
        string_list (List[str]): The list of strings from which to remove the word.

    Returns:
        List[str]: The resulting list of strings.
    """
    return [string.replace(word_to_remove, "") for string in string_list]

def compare_topics(df1: pd.DataFrame, df2: pd.DataFrame, topic_num: int) -> Dict[str,int]:
    """Compare the Topic Columns in two Dataframes and return a results dict.

    Args:
        df1 (pd.DataFrame): Input Dataframe 1
        df2 (pd.DataFrame): Input Dataframe 2
        topic_num (int): The Topic Number for which we calculate the results.

    Returns:
        dict: A dictionary containing changes from df1 to df2 : 
                a. Total Changes : Total Documents that changed it's topic assignment.
                b. Total Same : Total Documents that remained in the same topics. 
                c. Topic to Noise : Total Documents in "topic_num" that changed to noise. 
                d. All to Noise : Total Documents that changed to noise. 
                e. Topic Change : Total Documents in "topic_num" that changed to some other topic. 
                f. Topic Same : Total Documents in "topic_num" that remained in the same topic. 
    """
    ## 1/ Count the number of elements that changed between the two dataframes in Topic column
    # Select only the "Topic" column from each dataframe
    topics1 = df1["Topic"]
    topics2 = df2["Topic"]
    # Compare the two columns and count the number of changes
    changes = (topics1 != topics2).sum()
    # Count the number of elements that remained the same
    same = (topics1 == topics2).sum()

    ## 2/ Count the number of elements that changed to -1 or noise from "topic_num" topic.
    # Find rows where "Topic" changed from a non-negative value to -1
    changed_rows = (df1["Topic"] == topic_num) & (df2["Topic"] == -1)
    changed_rows_2 = (df1["Topic"] >= 0) & (df2["Topic"] == -1)
    # Extract the rows that satisfy the condition
    changed_rows_df = df1[changed_rows]
    changed_rows_df_2 = df1[changed_rows_2]
    # Get the number of rows that changed
    top2noise = len(changed_rows_df)
    all2noise = len(changed_rows_df_2)

    ## 3/ Check number of changes in topic constricted to "topic_num" topic.
    # Select rows in df1 where "Topic" is equal to the given topic
    rows_with_given_topic_df1 = df1[df1["Topic"] == topic_num]
    # Find the corresponding rows in df2
    corresponding_rows_df2 = df2.loc[rows_with_given_topic_df1.index]
    # Count the number of rows where the "Topic" value changed
    num_changed_rows = (
        rows_with_given_topic_df1["Topic"] != corresponding_rows_df2["Topic"]
    ).sum()
    # Count the number of rows where the "Topic" value remained the same
    num_same_rows = (
        rows_with_given_topic_df1["Topic"] == corresponding_rows_df2["Topic"]
    ).sum()

    results = {
        "total_changes": changes,
        "total_same": same,
        "topic_to_noise": top2noise,
        "all_to_noise": all2noise,
        "topic_change": num_changed_rows,
        "topic_same": num_same_rows,
    }

    return results

def raw_centroid_checks( docs: List[str], k: int, model: int) -> pd.DataFrame:
    """
    Take as input a list of documents, a speicific number of topics to run the check for,
    and the model. 
    1. Run a basic bertopic model to fit_transform the dataset.
    2. For each topic in 1 to k :
        2.1 
    """
    final_ablation_mappings = {}
    anchor_topic_model = load_model(model)
    topics, probs = anchor_topic_model.fit_transform(docs)
    topic_list = anchor_topic_model.get_topic_info()
    c_tf_idf_mappings = anchor_topic_model.topic_representations_
    c_tf_idf_mappings = convert_ctfidf(c_tf_idf_mappings)

    # forming doc -> topic pairing
    df_basic_mapping = pd.DataFrame({"Document": docs, "Topic": topics})
    
    for topic_i in tqdm(range(k)):
        ablation_mappings = {}
        for word in c_tf_idf_mappings[topic_i].keys():
            new_docs = remove_word_from_list(word, docs)
            new_topics, probs = anchor_topic_model.transform(new_docs)
            df_new_mapping = pd.DataFrame({"Document": docs, "Topic": new_topics})
            ablation_mappings[word] =  df_new_mapping
        final_ablation_mappings[f"Topic_{topic_i}"] = ablation_mappings

    return final_ablation_mappings,c_tf_idf_mappings,df_basic_mapping,topic_list

def raw_centroid_checks_cumulative(docs: List[str], k: int, model : int) -> pd.DataFrame:
    """
    Take as input a list of topics and the initial documents, perturbs the documents
    by removing one topic word after another and repeats the modeling to find if the
    topic changes.
    """
    ablation_mappings = {}
    anchor_topic_model = load_model(model)
    topics, probs = anchor_topic_model.fit_transform(docs)
    topic_list = anchor_topic_model.get_topic_info()["Representation"]
    c_tf_idf_mappings = anchor_topic_model.topic_representations_

    # forming doc -> topic pairing
    df_basic_mapping = pd.DataFrame({"Document": docs, "Topic": topics})

    new_docs = docs # Initialize new_docs with the original documents

    for  word in topic_list[k+1]: 
        new_docs = remove_word_from_list(word, new_docs)

        new_topics, probs = anchor_topic_model.transform(new_docs)
        df_new_mapping = pd.DataFrame({"Document": new_docs, "Topic": new_topics})

        ablation_mappings[word] = df_new_mapping

    return ablation_mappings, c_tf_idf_mappings, df_basic_mapping, topic_list

def save_raw(data,path) -> None:
    path = path+"/Temporary_Results/Topic_Results/"
    for topic in range(len(data.keys())):
        top_data = data[f"Topic_{topic}"]
        try: 
            os.makedirs(path+f"/Topic_{topic}") 
        except : 
            pass
        for word in top_data.keys():
            top_data[word].to_csv(path_or_buf=path+f"/Topic_{topic}/{word}.csv",
                                  columns=["Topic"],
                                  index=False,
                                  header=False
                                  )

def load_raw(path,topic_num) -> None:
    loaded_data = {}
    path = path + "/Temporary_Results/Topic_Results"
    for topic in range(topic_num):
        topic_path = path + f"/Topic_{topic}"
        if os.path.exists(topic_path):
            topic_data = {}
            for file_name in os.listdir(topic_path):
                word = os.path.splitext(file_name)[0]
                file_path = topic_path +"/"+ file_name
                topic_df = pd.read_csv(file_path, names=["Topic"])
                topic_data[word] = topic_df
            loaded_data[f"Topic_{topic}"] = topic_data
    return loaded_data

def save_base(c_tf_idf_mappings,df_basic_mapping,topic_list,path) -> None:
    # save c_tf_idf_mappings
    path = path + "Temporary_Results/Base_Results/"
    with open(path + "/ctf_idf_mappings.json", 'w') as json_file:
        json.dump(c_tf_idf_mappings, json_file,cls=NpEncoder)
    df_basic_mapping.to_csv(path+"/df_basic_mapping.csv")
    topic_list.to_csv(path+"/base.csv",columns=["Representation","Count"])

def load_model(model:int) :
    """Load the corresponding model based on the model number

    Args:
        model (int): ranging from 1 - 4 
    """
    if model == 1 :
        return BERTopic()
    elif model == 2 :
        representation_model = KeyBERTInspired()
        return BERTopic(representation_model=representation_model)
    elif model == 3 :
        representation_model = PartOfSpeech("en_core_web_sm")
        return BERTopic(representation_model=representation_model)
    else : 
        representation_model = MaximalMarginalRelevance(diversity=0.3)
        return BERTopic(representation_model=representation_model)

def dump_comprehensiveness_results(docs: List[str],k:int,path:str,model:int):
    """Runs comprehensiveness checks for each of the top k topics formed by Bertopic on the given 
    docs and saves the results in the given path, creating two folders : 
        - Temporary_Results :
            - Base_Results : Containing the base Bertopic Model Results.
            - Topic_Results : Containing the ablation results per topic.
        - Processed_Results :
            - ...

    Args:
        docs (List[str]): The given input documents.
        k (int): The top k topics to get the details for.
        path (str): The path to store the results. 
    """
    ablation_top_k_topics = {}

    ablation_top_k_topics,c_tf_idf_mappings,df_basic_mapping,topic_list = raw_comprehensiveness_checks(docs,k,model)
    print("========Comprehensiveness Ablation Tests done========")

    save_base(c_tf_idf_mappings,df_basic_mapping,topic_list,path) # save base results
    save_raw(ablation_top_k_topics,path) # save raw topic results 

def dump_comprehensiveness_results_cumulative(docs: List[str],k:int,path:str,model:int):
    """Runs comprehensiveness checks for each of the top k topics formed by Bertopic on the given 
    docs and saves the results in the given path, creating two folders : 
        - Temporary_Results :
            - Base_Results : Containing the base Bertopic Model Results.
            - Topic_Results : Containing the ablation results per topic.
        - Processed_Results :
            - ...

    Args:
        docs (List[str]): The given input documents.
        k (int): The top k topics to get the details for.
        path (str): The path to store the results. 
    """
    ablation_top_k_topics = {}

    for topic_i in tqdm(range(k)):
        ablation_top_k_topics[f"Topic_{topic_i}"],c_tf_idf_mappings,df_basic_mapping,topic_list = raw_comprehensiveness_checks_cumulative(docs,topic_i,model)
    print("========Comprehensiveness Ablation Tests done========")

    save_base(c_tf_idf_mappings,df_basic_mapping,topic_list,path) # save base results
    save_raw(ablation_top_k_topics,path) # save raw topic results 