import streamlit as st
import subprocess
import datetime
import torch
import wave
import contextlib
import numpy as np
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from pydub import AudioSegment
import struct
import whisper
import os
from openai import OpenAI
import time

OPENAI_API_KEY = os.getenv("OPEN_AI_TOKEN")
print(OPENAI_API_KEY)

# Function to perform speaker diarization and transcription
def speaker_diarization(path, model_size='large', num_speakers=2):
    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        device=torch.device("cpu"))

    if path[-3:] != 'wav':
        subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'], shell=True)
        path = 'audio.wav'

    with open(path, 'rb') as f1:
        header_beginning = f1.read(0x18)
        num_channels, = struct.unpack_from('<H', header_beginning, 0x16)

    if num_channels > 1:
        out_path = 'audio_1_channel_converted.wav'
        sound = AudioSegment.from_wav(path)
        sound = sound.set_channels(1)
        sound.export(out_path, format="wav")
        path = out_path

    model = whisper.load_model(model_size)

    start_time = time.time()
    print("Transcribing starts..")
    result = model.transcribe(path)

    load_time = time.time() - start_time
    print(f"Model tTranscribed in {load_time:.2f} seconds")
    segments = result["segments"]

    with contextlib.closing(wave.open(path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    audio = Audio()

    def segment_embedding(segment):
        start = segment["start"]
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(path, clip)
        return embedding_model(waveform[None])

    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment)

    embeddings = np.nan_to_num(embeddings)
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_

    for i in range(len(segments)):
        segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

    with open("transcript.txt", "w") as f:
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                f.write("\n" + segment["speaker"] + ' ' + str(datetime.timedelta(seconds=round(segment["start"]))) + '\n')
            f.write(segment["text"][1:] + ' ')

def read_transcript(file_path):
    with open(file_path, 'r') as file:
        transcript = file.read()
    return transcript

def create_prompt(transcript, current_date="12-12-2023"):
    prompt = f"""
    Imagine you are a MoM generator from the following transcript. Take the below conversation from a meeting and generate the minutes of the meeting and create a detailed table containing the list of tasks assigned to each person, the status of each task, and the deadlines. Write dates as well in the output table. Today is {current_date}. Identify the speaker names from the meeting transcript.

    {transcript}
    """
    return prompt

def generate_mom(prompt):
    # client = OpenAI(api_key=os.getenv('OPEN_AI_TOKEN'))
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": prompt}], max_tokens=1000)
    
    return response.choices[0].message.content

def main():
    st.title("Minutes of Meeting Generator - WHISPER")

    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

    if uploaded_file is not None:
        with open("temp_audio." + uploaded_file.name.split(".")[-1], "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Generate MoM"):
            with st.spinner('Processing...'):
                speaker_diarization("temp_audio." + uploaded_file.name.split(".")[-1])
                transcript = read_transcript("transcript.txt")
                prompt = create_prompt(transcript)
                mom = generate_mom(prompt)

                st.subheader("Minutes of Meeting")
                st.write(mom)

                with open("minutes_of_meeting.txt", "w") as f:
                    f.write(mom)

                st.download_button(
                    label="Download MoM",
                    data=mom,
                    file_name="minutes_of_meeting.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
