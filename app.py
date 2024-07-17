import speech_recognition as sr
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import openai
import os
import streamlit as st
from pathlib import Path


recognizer = sr.Recognizer()


def video_to_audio(video_path):
        video = VideoFileClip(video_path)
        audio_path = video_path.rsplit('.', 1)[0] + ".wav"
        video.audio.write_audiofile(audio_path)
        return audio_path



def recognize_speech_from_audio(audio_path):
     audio = AudioSegment.from_file(audio_path)
     audio.export(audio_path, format="wav")
     with sr.AudioFile(audio_path) as source:
         audio_data = recognizer.record(source)
         text = recognizer.recognize_google(audio_data)
     return text

KEY = os.getenv('OPENAI_API_KEY')


from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI


TEMPLATE="""
You are a Helpful Assistant. Summarize the text given below.
input_text: {input_text}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

from langchain.chains import LLMChain

medical_help_prompt = PromptTemplate(
      input_variables=["input_text"],
      template=TEMPLATE
)

llm=ChatOpenAI(openai_api_key
               =KEY,model_name="gpt-3.5-turbo", temperature=0.5)

from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

quiz_chain=LLMChain(llm=llm, prompt=medical_help_prompt, output_key='answer', verbose=True)

def summarize_text(text):
     final_result = quiz_chain.run(
     {
         "input_text":text
     }
     )
     return final_result


def summarize(file_path):
     temp_dir = "temp"
     os.makedirs(temp_dir, exist_ok=True)

     try:
         file_path_str  = str(file_path)
         if file_path_str .endswith((".mp4", ".mkv", ".avi")):
             audio_path = video_to_audio(file_path_str)
         elif file_path_str.endswith((".wav", ".m4a")):
             audio_path = file_path_str
         else:
             raise ValueError("Unsupported file type")

         text = recognize_speech_from_audio(audio_path)
         summary = summarize_text(text)
         return {"original_text": text, "summary": summary}
     except Exception as e:
         return {"error": str(e)}


# Streamlit application
def main():
     st.title("Video/Audio Summarizer")
     st.write("Upload a video or audio file, and get the original text and its summary.")

     uploaded_file = st.file_uploader("Choose a video or audio file", type=["mp4", "mkv", "avi", "wav", "m4a"])

     if uploaded_file is not None:
         file_path = Path("temp") / uploaded_file.name
         with open(file_path, "wb") as f:
             f.write(uploaded_file.getbuffer())

         result = summarize(file_path)

         if "error" in result:
             st.error(result["error"])
         else:
             st.subheader("Original Text")
             st.write(result["original_text"])

             st.subheader("Summary")
             st.write(result["summary"])

if __name__ == "__main__":
     main()