import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Load the NLP model (you can choose any model you prefer, here we use GPT-2)
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForCausalLM.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
nlp = pipeline("text-generation", model=model, tokenizer=tokenizer, truncation=True, pad_token_id=tokenizer.eos_token_id)

# Function to generate plotting code from the prompt using NLP model
def generate_plotting_code(prompt):
    result = nlp(prompt, max_length=50,truncation=True)[0]['generated_text']
    print(result)
    return result

# Function to read Excel data and plot the graph based on the generated code
def plot_graph(prompt):
    # Read the Excel file
    try:
        df = pd.read_excel(r"C:\Users\chris\OneDrive\Desktop\JAFFA-AI\Excel Plotting Agent\test\dummy_excel.xlsx")
    except Exception as e:
        return str(e), "Error reading the Excel file."
    
    # Generate the plotting code from the prompt
    plotting_code = generate_plotting_code(prompt)
    
    # Prepare the plotting environment
    fig, ax = plt.subplots()

    # Execute the generated plotting code
    try:
        exec(plotting_code)
        plt.savefig("plot.png")
        plt.close()
        return "plot.png", plotting_code
    except Exception as e:
        return str(e), plotting_code

# Gradio interface
interface = gr.Interface(
    fn=plot_graph,
    inputs=["text"],
    outputs=["image", "text"],
    title="Excel Plotting Agent",
    description="provide a prompt to plot a graph. The agent will interpret the prompt and generate the plot.",
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()
