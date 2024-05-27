import os
import streamlit as st
import logging
from dotenv import load_dotenv
from datetime import datetime
import httpx
import json
from deepgram import DeepgramClient, DeepgramClientOptions, PrerecordedOptions
from openai import OpenAI

# Load environment variables
load_dotenv()

# Constants
API_KEY = os.getenv("DG_API_KEY")
OPENAI_API_KEY = os.getenv("OPEN_AI_TOKEN")
MIMETYPE = 'mp3'
TAG = 'SPEAKER '
SEPARATOR = '--------------------------'

# Initialize Deepgram client
deepgram_client = DeepgramClient(API_KEY, DeepgramClientOptions(verbose=logging.DEBUG))

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Function to transcribe an audio file
def transcribe_audio(file):
    buffer_data = file.read()
    payload = {"buffer": buffer_data}
    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        utterances=True,
        punctuate=True,
        diarize=True,
    )

    response = deepgram_client.listen.prerecorded.v("1").transcribe_file(payload, options, timeout=httpx.Timeout(300.0, connect=10.0))
    return response.to_dict()

# Function to create a transcript from JSON response
def create_transcript(response):
    lines = []
    words = response["results"]["channels"][0]["alternatives"][0]["words"]
    curr_speaker = 0
    curr_line = ''
    for word_struct in words:
        word_speaker = word_struct["speaker"]
        word = word_struct["punctuated_word"]
        if word_speaker == curr_speaker:
            curr_line += ' ' + word
        else:
            tag = TAG + str(curr_speaker) + ':'
            full_line = tag + curr_line + '\n'
            curr_speaker = word_speaker
            lines.append(full_line)
            curr_line = ' ' + word
    lines.append(TAG + str(curr_speaker) + ':' + curr_line)
    return '\n'.join(lines)

# Function to create a prompt for the MoM generator
def create_prompt(transcript):
    current_date = datetime.now().strftime("%d-%m-%Y")
    prompt = f"""
    Imagine you are a MoM generator from the following transcript. Take the below conversation from a meeting and generate the minutes of the meeting and create a detailed table containing the list of tasks assigned to each person, the status of each task, and the deadlines. Write dates as well in the output table. Today is {current_date}. Identify the speaker names from the meeting transcript.

    {transcript}
    """
    return prompt

# Function to generate MoM using OpenAI's GPT
def generate_mom(prompt):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": prompt}], 
        max_tokens=1000
    )
    return response.choices[0].message.content

# Streamlit app
st.set_page_config(page_title="Minutes of Meeting Generator", page_icon="👄")

# logo and header -------------------------------------------------

st.text("")
st.image(
    "https://www.kapwing.com/resources/content/images/size/w1200/2024/02/how-to-transcribe-interviews.webp",
    width=205,
)

st.title("Minutes of Meeting Generator using Deepgram and GPT")

st.write(
    """  
-   Upload a mp3 file, transcribe and diarize it, then export it to a text file!
-   Use cases: call centres, team meetings, training videos, school calls etc.
	    """
)

st.text("")

# Initialize variables to store the transcript and MoM
transcript = None
mom = None

with st.form(key="my_form"):
    # File upload
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3"])

    st.info(
        f"""
        👆 Upload a .mp3 file. Try a sample: [Sample 01](https://github.com/arjunaju123/MoM-generation-app-using-Deepgram-and-GPT/blob/master/test_audio.mp3?raw=true) | [Sample 02](https://github.com/arjunaju123/MoM-generation-app-using-Deepgram-and-GPT/blob/master/test_audio.mp3?raw=true)
        """
    )

    submit_button = st.form_submit_button(label="Generate Minutes of Meeting")
    if submit_button and uploaded_file is not None:
        with st.spinner("Transcribing and generating MoM..."):
            try:
                # Transcribe the audio file
                response = transcribe_audio(uploaded_file)
                
                # Create the transcript
                transcript = create_transcript(response)
                
                # Create prompt for MoM generation
                prompt = create_prompt(transcript)
                
                # Generate MoM
                mom = generate_mom(prompt)
            except Exception as e:
                st.error(f"Error: {e}")

# Display the audio player, transcript, and MoM if available
if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")

if transcript:
    st.subheader("Diarized Transcript")
    st.info(transcript)

if mom:
    st.subheader("Minutes of Meeting")
    st.info(mom)
    st.download_button("Download Minutes of Meeting", mom, "minutes_of_meeting.txt", "text/plain")
