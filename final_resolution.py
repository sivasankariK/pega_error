import pandas as pd
import json
import openai
import os
import gradio as gr
openai_api_key = ""

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the dataset from a given file path."""
    return pd.read_csv(file_path, index_col=False)

def combine_columns_to_prompt(dataset: pd.DataFrame) -> pd.DataFrame:
    """Combine 'issuename' and 'issue description' columns into 'prompt'."""
    dataset['prompt'] = dataset.apply(lambda row: f"Issue Name: {row['issuename']}\nDescription: {row['issue description']}", axis=1)
    return dataset

def filter_prompt_column(dataset: pd.DataFrame) -> pd.DataFrame:
    """Filter the dataset to only retain the 'prompt' column."""
    return dataset.filter(["prompt"])
def extract_resolution(dataset:pd.DataFrame)-> pd.DataFrame:
    return dataset.filter(["issue resolution"])

def reformat_jsonl(file_path: str) -> str:
    """Reformat a JSONL file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    json_objects = [json.loads(line) for line in lines]
    new_content = []
    for obj in json_objects:
        for key in obj["prompt"]:
            new_content.append(json.dumps({
                "prompt": obj["prompt"][key]
            }))

    new_file_path = file_path.replace(".jsonl", "_reformatted.jsonl")
    with open(new_file_path, "w") as file:
        for line in new_content:
            file.write(line + "\n")

    return new_file_path

def transcribe_audio(file_path: str) -> str:
    """Transcribe the audio file using OpenAI's Whisper API."""
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript.text

def save_to_file(content: str, filename: str) -> None:
    """Save the given content to a specified file."""
    with open(filename, 'w') as file:
        file.write(content)
    print(f"Content saved to {filename}")

def read_from_file(filename: str) -> str:
    """Read the content of the specified file."""
    with open(filename, 'r') as file:
        content = file.read()
    return content    


def get_questions(prompt, model_choice,issue_resolution):
    if isinstance(prompt, list):
        prompt = prompt[0]
    source_document = "D:\pega_error/transcript_output.txt"
    source_document = read_from_file(source_document)
    if model_choice == "text-ada-001":
        response = openai.Completion.create(
            engine="text-ada-001",
            prompt=f"""use the provided source document and Your task is to answer the question using only the provided document and to cite the passage(s) of the document used to answer the question.
                You will be provided with Error message services that require troubleshooting the error messages. Your task is to find the corrective action of the error messages from the source documents\n\nText: {prompt}\n\nAnalyzing the source document{source_document} to find the corrective action:\n1.""",
            temperature=0,
            max_tokens=257,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"]
        )
        return response['choices'][0]['text']
    elif model_choice == "text-davinci-003":
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"""use the provided source document and Your task is to answer the question using only the provided document and to cite the passage(s) of the document used to answer the question.
                You will be provided with Error message services that require troubleshooting the error messages. Your task is to find the corrective action of the error messages from the source documents\n\nText: {prompt}\n\nAnalyzing the source document {source_document}to find the corrective action:\n1.""",
            temperature=0,
            max_tokens=257,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"]
        )
        return response['choices'][0]['text']
      
    else:
        response = openai.ChatCompletion.create(
            model=model_choice,  # Use the model_choice parameter to specify the model
        #     messages=[
        #         {
        #         "role": "system",
        #         "content": """use the provided source document and Your task is to answer the question using only the provided document and to cite the passage(s) of the document used to answer the question.
        #         You will be provided with Error message services that require troubleshooting the error messages. Your task is to find the corrective action of the error messages from the source documents"""
        #         },
        #         {
        #         "role": "user",
        #         "content": f"Question: {prompt}\nSource Document: {source_document}"
        #         },
        #         {
        #         "role": "assistant",
        #         "content": "Analyzing the source document to find the corrective action",
        #         }
        #     ]
        # )
        # chat_response = response['choices'][0]['message']['content']
        # similarity_metric = compare_responses(chat_response, issue_resolution)
    
        # return chat_response, issue_resolution, similarity_metric

            messages=[
                {
                "role": "system",
                "content": """First, conduct an independent analysis based on the {prompt} and formulate your own solution. Then, compare your solution with the provided {issue_resolution} and the {source_documents}to assess the accuracy of your solution. Please refrain from judging the correctness of your solution until you have completed your personal analysis. """
                },
                {
                "role": "user",
                "content": f"Question: {prompt}"
                },
                {
                "role": "assistant",
                # "content": "Analyzing the source document and to find the corrective action ",
                "content": "To find the corrective action based on {issue_resolution} ",
                
                }
            ]
        )
        chat_response = response['choices'][0]['message']['content']
        similarity_metric = compare_responses(chat_response, issue_resolution)
    
        return chat_response, issue_resolution, similarity_metric
        
        # messages=[
        #         {
        #         "role": "system",
        #         "content": """First, conduct an independent analysis based on the {prompt} and formulate your own solution. Then, compare your solution with the provided {issue_resolution} to assess the accuracy of your solution. Please refrain from judging the correctness of your solution until you have completed your personal analysis. """
        #         },
        #         {
        #         "role": "user",
        #         "content": f"Question: {prompt}\nSource Document: {source_document}"
        #         },
        #         {
        #         "role": "assistant",
        #         "content": "Analyzing the {source_document}\n{issue_resolution} and  to find the corrective action and similarity measure",
        #         }
        #     ]
        # )
        # chat_response = response['choices'][0]['message']['content']
        # similarity_metric = compare_responses(chat_response, issue_resolution)
    
        # return chat_response, issue_resolution, similarity_metric
def compare_responses(chat_response, issue_resolution):
    
    return chat_response == issue_resolution
if __name__ == "__main__":
    dataset = "D:/pega_error/data_pega.csv"
    dataset = load_dataset(dataset)
# Combine columns to create the 'prompt' column
    dataset = combine_columns_to_prompt(dataset)

# Filter the dataset to only retain the 'prompt' column
    filtered_dataset = filter_prompt_column(dataset)
    issue_resolution_column = dataset['issue resolution']
    model_choices = ["gpt-3.5-turbo","gpt-3.5-turbo-0301","gpt-3.5-turbo-0613","gpt-4","gpt-4-0613","gpt-4-0314"]
    prompts_list = filtered_dataset['prompt'].tolist()
    issue_resolutions = dataset['issue resolution'].unique().tolist()
# Iterate over each model choice
for model in model_choices:
        print(f"Fetching responses using model: {model}")
    
    # Iterate over the 'prompt' column and call the get_questions function for each prompt
for (index, row), issue_resolution in zip(filtered_dataset.iterrows(), issue_resolution_column):
        prompt = row['prompt']
        response = get_questions(prompt, model, issue_resolution)
       
        # print(response)
        # print("\n")    

# gr.Interface(fn=get_questions, 
#                  inputs=[gr.Dropdown(choices=prompts_list, label="Choose a Prompt"),gr.Radio(choices=["gpt-3.5-turbo","gpt-3.5-turbo-0301","gpt-3.5-turbo-0613","gpt-4","gpt-4-0613","gpt-4-0314","text-ada-001", "text-davinci-003"], label="Choose a Model")],
#                  outputs=[gr.components.Textbox(label="Chat Response"), gr.Dropdown(choices=issue_list,label="Issue Resolution"), gr.components.Textbox(label="Similarity Metric")]).launch()
gr.Interface(fn=get_questions, 
                 inputs=[gr.Dropdown(choices=prompts_list, label="Choose a Prompt"),
                         gr.Radio(choices=["gpt-3.5-turbo","gpt-3.5-turbo-0301","gpt-3.5-turbo-0613","gpt-4","gpt-4-0613","gpt-4-0314","text-ada-001", "text-davinci-003"], label="Choose a Model"),
                         gr.Dropdown(choices=issue_resolutions, label="Choose Issue Resolution")],
                 outputs=[gr.components.Textbox(label="Chat Response"), 
                          gr.components.Textbox(label="Issue Resolution"), 
                          gr.components.Textbox(label="Similarity Metric")]).launch()




