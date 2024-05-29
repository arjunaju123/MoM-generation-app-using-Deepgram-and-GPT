import os
import streamlit as st
import logging
from dotenv import load_dotenv
from datetime import datetime
import httpx
import json
from deepgram import DeepgramClient, DeepgramClientOptions, PrerecordedOptions
from openai import OpenAI
import time

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

# Function to translate text using OpenAI's GPT
def translate_text(text, target_language):
    if(target_language!='english'):
        sub_prompt =f' Translate the following diarized output to {target_language}'
    else:
        sub_prompt =''
 
    print("text is ",text)
    #prompt = f"Translate the following text to {target_language}:\n\n{text}.Instead of 'SPEAKER 0' and 'SPEAKER 1' in text give names Gokul and Avinash. All words should be generated in provided language - {target_language} only"
    prompt = sub_prompt+f"\n{text}"+f"\n\nThis is the output text having only 2 speakers from a diarization model .Find the person names from the output text given here and replace the speaker ids like SPEAKER 0,SPEAKER 1 etc with corresponding person names.For example: Gokul: 'Okay. Yeah. Hi. I'm Gokul, and I'm I'm into the data science team from Experion.'\n\n\n'Avinash: 'Hi. I'm Avinash, and I'm also in the data science team of Experian.' Generate complete words in {target_language}."
    
    print("prompt is",prompt)
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )
    return response.choices[0].message.content

# Function to create a prompt for the MoM generator
def create_prompt(transcript, language='english'):
    current_date = datetime.now().strftime("%d-%m-%Y")
    prompt = f"""
    You are a MoM generator from the following transcript. Take the below conversation from a meeting and generate the minutes of the meeting and create a detailed table containing the list of tasks assigned to each person, the status of each task, and the deadlines. Write dates as well in the output table. Today is {current_date}. Identify the speaker names from the meeting transcript.
    
    Generate the Minutes of Meeting in {language} only.

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

# Center the image using HTML and CSS
st.markdown(
    """
    <style>
    .center {
        display: flex;
        justify-content: center;
    }
    </style>
    <div class="center">
        <img src="data:image/jpeg;base64,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" width="305">
    </div>
    """,
    unsafe_allow_html=True
)


st.title("Enhancing Conversations with AI")

st.write(
    """  
-   Automatically Generate Minutes of Meetings from Meeting Recordings
-   View the Minutes of Meetings in the preferred language
    """
)

st.text("")

# Initialize variables to store the transcript and MoM
transcript = None
translated_transcript = None
mom = None

with st.form(key="my_form"):
    # File upload
    uploaded_file = st.file_uploader("Upload Audio", type=["mp3","wav"])

    # Language selection
    language = st.selectbox("Select the language for MoM:", ["English", "Japanese"])

    st.info(
        f"""
        👆 Upload a .mp3 / .wav file. Try a sample: [Sample 01](https://github.com/arjunaju123/MoM-generation-app-using-Deepgram-and-GPT/blob/master/test_audio.mp3?raw=true) | [Sample 02](https://github.com/arjunaju123/MoM-generation-app-using-Deepgram-and-GPT/blob/master/test_audio.wav?raw=true)
        """
    )

    submit_button = st.form_submit_button(label="Generate Minutes of Meeting")
    if submit_button and uploaded_file is not None:
        with st.status("Transcribing and generating MoM...",expanded=True) as status:
            try:
                # Transcribe the audio file
                start_time = time.time()
                st.write("Transcribing audio...")
                response = transcribe_audio(uploaded_file)
                
                # Create the transcript
                transcript = create_transcript(response)
                transcribe_time = time.time() - start_time
                # Translate the transcript if the selected language is not English
                if language != 'english':
                    translated_transcript = translate_text(transcript, language)
                else:
                    translated_transcript = transcript

                st.write(f"Time taken to transcribe: {transcribe_time:.2f} seconds")

                # Create prompt for MoM generation
                prompt = create_prompt(translated_transcript, language)
                
                st.write("Generating MoM...")
                start_time = time.time()
                # Generate MoM
                mom = generate_mom(prompt)
                generate_mom_time = time.time() - start_time
                st.write(f"Time taken to generate MoM: {generate_mom_time:.2f} seconds")

                status.update(label="Transcription and Generation Completed!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Error: {e}")

# Display the audio player, translated transcript, and MoM if available
if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")

if translated_transcript:
    st.subheader(f"Diarized Transcript in {language.capitalize()}")
    st.info(translated_transcript)

if mom:
    st.subheader("Minutes of Meeting")
    st.info(mom)
    st.download_button("Download Minutes of Meeting", mom, "minutes_of_meeting.txt", "text/plain")
