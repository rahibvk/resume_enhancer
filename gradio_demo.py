# Import necessary packages
from ibm_watson_machine_learning.foundation_models import Model
import gradio as gr

# Set credentials to use the model
my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

# Model and project settings
model_id = "meta-llama/llama-2-70b-chat"  # Directly specifying the LLAMA2 model
project_id = "skills-network"  # Specifying project_id as provided
space_id = None
verify = False
gen_parms = {
        "max_new_tokens": 900,
        "temperature": 0.1
}

# Initialize the model
model = Model(model_id, my_credentials, gen_parms, project_id, space_id, verify)

# Function to generate a response from the model
def generate_response(prompt_txt):
    generated_response = model.generate(prompt_txt)

    # Extract and return the generated text
    generated_text = generated_response["results"][0]["generated_text"]
    return generated_text

# Create Gradio interface
chat_application = gr.Interface(
    fn=generate_response,
    allow_flagging="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Watsonx.ai Chatbot",
    description="Ask any question and the chatbot will try to answer."
)

# Launch the app
chat_application.launch()