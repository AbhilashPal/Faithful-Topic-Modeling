import gradio as gr
import seaborn as sns
import matplotlib.pyplot as plt
from src.gradio_app.create_graphs import create_graph

def plot(metric, dataset, model, topic, plot_type):

    model_string = {
        "cTF-IDF" : "model_1","Keybert":"model_2","PoS":"model_3","MMR":"model_4",
        "LDA":"lda_model","Randomization":"randomization"
    }
    base = f"result/{metric}/{dataset}/{model_string[model]}"
    plot_map = {
        "Total_Changes":1, "Topic_Changes":2, "Topic_To_Noise":3, "All_To_Noise":4, "Topic_Same" : 5
    } 
    create_graph(base,topic,plot_map[plot_type],model)
    return (base+f"/Processed_Results/graphs/Topic_{topic}/{plot_type}.png")
    

# Create the Gradio interface
inputs = [
    gr.Dropdown(["comprehensiveness","sufficiency"], label="Metric",value="comprehensiveness"),
    gr.Dropdown(["nyt", "20newsgroup", "wiki"], label="Dataset",value="20newsgroup"),
    gr.Dropdown(["cTF-IDF","Keybert","PoS","MMR","LDA","Randomization"], label="Model",value="Randomization"),
    gr.Slider(minimum=1, maximum=100, step=1, label="Topic",value=1),
    gr.Dropdown(["Total_Changes", "Topic_Changes", "Topic_To_Noise", "All_To_Noise","Topic_Same"], label="Plot Type",value="Topic_Changes")
]
outputs = gr.Image(label="Plot")
gr.Interface(fn=plot, inputs=inputs, outputs=outputs, theme="soft").launch(share=True)
