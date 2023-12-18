import gradio as gr
import seaborn as sns
import matplotlib.pyplot as plt
from src.gradio_app.create_graphs import create_graph_comprehensiveness

def plot(dataset, topic, plot_type):
    base = f"results/keybert/{dataset}"
    plot_map = {
        "Total_Changes":1, "Topic_Changes":2, "Topic_To_Noise":3, "All_To_Noise":4, "Centroid_Movement" : 5
    } 
    create_graph_comprehensiveness(base,topic,plot_map[plot_type])
    return (base+f"/Processed_Results/graphs/Topic_{topic}/{plot_type}.png")
    

# Create the Gradio interface
inputs = [
    gr.Dropdown(["nyt", "20newsgroup", "wiki"], label="Dataset"),
    gr.Slider(minimum=1, maximum=100, step=1, label="Topic"),
    gr.Dropdown(["Total_Changes", "Topic_Changes", "Topic_To_Noise", "All_To_Noise", "Centroid_Movement"], label="Plot Type")
]
outputs = gr.Image(label="Plot")
gr.Interface(fn=plot, inputs=inputs, outputs=outputs, theme="soft").launch(share=True)
