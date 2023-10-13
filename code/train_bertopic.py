import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import joblib

def train_and_save_bertopic_model(documents, num_topics=10, language='english', model_path='bertopic_model.pkl'):
    """
    Train a BERTopic model on a list of documents and save the model to a file.

    Parameters:
    - documents (list): A list of text documents for topic modeling.
    - num_topics (int): The number of topics to extract.
    - language (str): The language of the documents. Use a BERT model that supports this language.
    - model_path (str): The path to save the trained model.

    Returns:
    - model: The trained BERTopic model.
    - topics: The generated topics and their top words.
    - topic_sizes: The sizes of each topic.
    """
    # Load a pre-trained BERT model suitable for the given language
    if language == 'english':
        model_name = 'bert-base-nli-mean-tokens'
    elif language == 'german':
        model_name = 'bert-base-nli-stsb-mean-tokens'
    # You can specify more models according to your language.

    sentence_model = SentenceTransformer(model_name)

    # Create a BERTopic model
    model = BERTopic(language=language, calculate_probabilities=True)

    # Fit the model on the documents
    topics, topic_sizes = model.fit_transform(documents)

    # Save the model to a file using joblib
    joblib.dump(model, model_path)

    return model, topics, topic_sizes

def load_bertopic_model(model_path):
    """
    Load a saved BERTopic model from a file.

    Parameters:
    - model_path (str): The path to the saved BERTopic model.

    Returns:
    - model: The loaded BERTopic model.
    """
    model = joblib.load(model_path)
    return model
    
# Example usage:
if __name__ == "__main__":
    # Replace 'documents' with your own list of text documents.
    documents = [
        "This is the first document about BERTopic.",
        "BERTopic is a great library for topic modeling.",
        "Topic modeling can help extract meaningful topics from text.",
        "I'm experimenting with BERTopic on my own dataset.",
    ]

    model, topics, topic_sizes = train_and_save_bertopic_model(documents, num_topics=5, language='english', model_path='bertopic_model.pkl')

    # Print the topics and their top words
    for topic_id, topic_words in topics.items():
        print(f"Topic {topic_id}: {', '.join(topic_words)}")

    # Print the topic sizes
    print("Topic Sizes:", topic_sizes)

