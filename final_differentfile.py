import os
from docx import Document
import fitz  # PyMuPDF
import gradio as gr
import openai
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import traceback
# Initialize the OpenAI API key
openai.api_key = ""
def read_file(file_path):
    _, ext = os.path.splitext(file_path.lower())
    
    try:
        if ext == '.txt':
            with open(file_path, 'r') as file:
                data = file.read()
        elif ext == '.pdf':
            doc = fitz.open(file_path)
            data = ""
            for page in doc:
                data += page.get_text()
        elif ext == '.docx':
            doc = Document(file_path)
            data = ""
            for para in doc.paragraphs:
                data += para.text
        elif ext == '.mp4':
            audio = AudioSegment.from_file(file_path)
            chunk_length = 30000  # Length of each chunk in milliseconds (30s)
            chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]

            data = ""
            for i, chunk in enumerate(chunks):
                chunk_file_path = f"chunk{i}.wav"
                chunk.export(chunk_file_path, format="wav")

                with open(chunk_file_path, "rb") as audio_file:
                    transcript = openai.Audio.transcribe("whisper-1", audio_file)
                data += transcript.text + " "
                
                # Delete the temporary chunk file
                os.remove(chunk_file_path)
        else:
            return f"Unsupported file extension: {ext}", None
        
        return data, ext
    except Exception as e:
        traceback.print_exc()
        return f"An error occurred: {e}", None



def load_dataset(data):
    """Load the dataset from a data frame."""
    return data

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


def get_questions(source_document, model_choice,prompt):
    if model_choice == "text-ada-001":
        response = openai.Completion.create(
            engine="text-ada-001",
            prompt=f"Question :{prompt} based on the question retrieve the extact answer from the source document  Source Document: {source_document}\n\nGenerate a response based on the source document:\n",
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
            prompt=f"Source Document: {source_document}\n\nGenerate a response based on the source document:\n",
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
            messages=[
                {
                "role": "system",
                "content":f"""You will be provided with a source document {source_document} and a question as {prompt}. Your task is to answer the question using only the provided document and to extract the passage of the document used to answer the question. If the document does not contain the information needed to answer this question then simply write: Insufficient information"""
                },
                
                {
                "role": "user",
                "content": f"Question: {prompt}"
                },
                {
                "role": "assistant",
                "content": "extract the information from the {source_document} "
                
                }
            ]
        )
        return response['choices'][0]['message']['content']
        
    
     

def process_input(file, model_choice,prompt):
    data, ext = read_file(file.name)
    if data is None:
        return str(ext)
    
    response = get_questions(data, model_choice,prompt)
    return response

if __name__ == "__main__":
    
    gr.Interface(
        fn=process_input, 
        inputs=[
            gr.components.File(label="Upload a File"), 
            gr.Radio(choices=["gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-0613", "gpt-4-0314", "text-ada-001", "text-davinci-003"], label="Choose a Model"),
            gr.components.Textbox(label="Enter a Prompt")
        ],
        outputs=[
            gr.components.Textbox(label="Chat Response"), 
            
        ]
    ).launch()
