# Minutes of Meeting Generator

This Streamlit application allows users to upload audio files (in mp3 or wav format), transcribe and diarize the audio using Deepgram, and then generate Minutes of Meeting (MoM) using OpenAI's GPT. The MoM and the diarized transcript can be generated in English or Japanese.

## Features

- Upload an audio file and transcribe it.
- Diarize the transcription (identify and separate speakers).
- Translate the transcript to Japanese (if selected).
- Generate Minutes of Meeting from the transcript.
- Download the generated Minutes of Meeting.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or later installed on your machine.
- Streamlit installed. You can install it using pip:
  ```bash
  pip install streamlit
  ```
- The required Python packages installed. You can install them using pip:
  ```bash
  pip install httpx python-dotenv deepgram-sdk openai
  ```

## Getting Started

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/arjunaju123/MoM-generation-app-using-Deepgram-and-GPT.git
cd mom-generator
```

### 2. Set Up Environment Variables

Create a `.env` file in the project directory and add your Deepgram API key and OpenAI API key:

```ini
DG_API_KEY=your_deepgram_api_key
OPEN_AI_TOKEN=your_openai_api_key
```

### 3. Run the Application

Run the Streamlit app using the following command:

```bash
streamlit run sample.py --server.enableXsrfProtection false
```

This command will start the Streamlit server and open the app in your default web browser.

## Usage

1. Upload an audio file (mp3 or wav format) using the file uploader.
2. Select the language for the Minutes of Meeting (English or Japanese) from the dropdown menu.
3. Click the "Generate Minutes of Meeting" button.
4. Wait for the transcription and MoM generation process to complete. The time taken for each step will be displayed.
5. The diarized transcript and generated Minutes of Meeting will be displayed on the page.
6. Optionally, download the generated Minutes of Meeting as a text file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Deepgram](https://www.deepgram.com) for providing the speech-to-text API.
- [OpenAI](https://www.openai.com) for providing the language generation API.
- [Streamlit](https://www.streamlit.io) for providing the framework to build this web application.
```

Replace `https://github.com/yourusername/mom-generator.git` with the actual URL of your repository if you have one. This README provides an overview of the project, setup instructions, and usage guidelines.