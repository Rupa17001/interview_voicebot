import streamlit as st
import os
from deepgram import DeepgramClient, SpeakOptions, PrerecordedOptions, FileSource
from dotenv import load_dotenv
import requests
import wave
import uuid
from audio_recorder_streamlit import audio_recorder 


load_dotenv()

deep_api = os.getenv("DEEPGRAM_API_KEY", "").strip()
hf = os.getenv("HF_Token", "").strip()
print(deep_api)
print(hf)

# Function to save raw bytes as WAV
def save_wav_file(file_path, wav_bytes):
    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setnchannels(1) 
        wav_file.setsampwidth(2)  
        wav_file.setframerate(44100)  
        wav_file.writeframes(wav_bytes)

def stt(audio_file):
    try:
        deepgram = DeepgramClient(api_key=deep_api)
        with open(audio_file, "rb") as file:
            buffer_data = file.read()
        payload: FileSource = {"buffer": buffer_data}
        options = PrerecordedOptions(model="nova-3", smart_format=True)
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        os.remove(audio_file)
        return response["results"]["channels"][0]["alternatives"][0]["transcript"]
    except Exception as e:
        st.error(f"Error in STT: {e}")
        return None

def query_mistral(payload):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    API_TOKEN = hf
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        print(response)
        if response.status_code != 200:
            return None
        
        data = response.json()
        generated_text = data[0]['generated_text']
        prompt_index = generated_text.find('[/INST]')
        if prompt_index != -1:
            generated_text = generated_text[prompt_index + len('[/INST]'):].strip()
        return generated_text
    except requests.exceptions.RequestException:
        return None
    except ValueError:
        return None

def using_mistral(query_text):
    prompt = f"""[INST] You are Rupa and you are sitting in an interview. You have to answer the question from given data only. DATA :Name : Rupa 
                    About : I am a software engineer who has been working in gen ai for last 2 years, till now I have got chance to build applications that are powered by AI, (question answer on personal data, summarize the document, fetch the entities from unstructures data)
                    skills that I have gained till now  : Language Python, Java(basic) Data Storage & Search Elasticsearch, MilvusDB, SQL, MongoDB Machine Learning Frameworks TensorFlow, PyTorch, Hugging Face, NLTK Deployment & Integration FastAPI, Docker, RESTful API Development IBM Cloud Generative AI Mistral, BART, BERT, LangChain, RAG (Retrieval-Augmented Generation), NLTK

                    Profile : Software Engineer (Generative AI)
                    Years of Experience : 2 years 3 months
                    SuperPower : 1.) Flashbacks – "Understanding the importance of reviewing past results to improve future decisions. In a recent POC, a solution was built but not 		optimized. Recognizing this, I suggested tuning and refining the application based on user feedback, which led to a visible improvement."
                            2.)Instant Insight & Clarity : Whether identifying pain points in an application or decoding a complex problem, bridging the gap between confusion and 		clarity comes naturally. Also, effortlessly breaking down intricate solutions into simple, digestible explanations ensures seamless communication  		across all levels."
                        they both combinly make a superpower called : MindMap

                    Area of growth : 1.) Technical : To build more scalable and efficient solutions as the technical world is a very dynamic space,
                            2.) Management : leadership and team work is really import as I'llget chance to work with different people and teams I can always work on my management 			skills
                            3.)System Design : that is something I realy want to work with to build a scalable solution 

                    misconception other have about me : That I am just a kid (beacause I look like one)
                    Pushing boundries and limits : I think hardwork and taking feedback from coworker works for me, whenever it is about pushing limits hardwork is important.(sometime I do watch motivational videos :)
        Question : {query_text}
        Answer in formal and polite manner and answer in limited words and answer must relate to question[/INST]"""
    
    max_new_tokens = 500
    return query_mistral({"parameters": {"max_new_tokens": max_new_tokens}, "inputs": prompt})

def tts(speak_text, filename):
    try:
        deepgram = DeepgramClient(api_key=deep_api)
        options = SpeakOptions(model="aura-asteria-en")
        response = deepgram.speak.rest.v("1").save(filename, {"text": speak_text}, options)
        return filename
    except Exception as e:
        st.error(f"Error in TTS: {e}")
        return None

def main():
    st.title("AI Interview Candidate Voice Bot")
    
    # Record audio using audio_recorder
    voice_input = audio_recorder(pause_threshold=2.5, sample_rate=44100)

    if voice_input:
        audio_path = f"input/audio_{uuid.uuid4().hex}.wav"
        with open(audio_path,"wb") as f:
            f.write(voice_input)
        st.success(f"Audio recorded: {audio_path}")
        
        st.write("Processing...")
        transcribed_text = stt(audio_path)
        
        if transcribed_text:
            st.write("**Transcribed Text:**", transcribed_text)
            response_text = using_mistral(transcribed_text)
            print(response_text)
            st.spinner("Processing...")
            if response_text:
                st.write("**AI Response:**", response_text)
                
                audio_response_path = "output/response.mp3"
                tts(response_text, audio_response_path)
                
                audio_file = open(audio_response_path, "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")

if __name__ == "__main__":
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    main()