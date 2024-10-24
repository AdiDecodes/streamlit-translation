import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from Model.Eng_Hin.__init__ import getEnModel
from Model.Hin_Eng.__init__ import getHiModel
import speech_recognition as sr
import time
import os

def load_model_and_tokenizer(model_path):
    if model_path == 'model/Eng_Hin':
        model = getEnModel()
    else:
        model = getHiModel()
    model = MarianMTModel.from_pretrained(model_path)
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    return model, tokenizer

en_to_hi_model_path = 'model/Eng_Hin'
hi_to_en_model_path = 'model/Hin_Eng'


en_to_hi_model, en_to_hi_tokenizer = load_model_and_tokenizer(en_to_hi_model_path)
hi_to_en_model, hi_to_en_tokenizer = load_model_and_tokenizer(hi_to_en_model_path)


st.title("Real-time Language Translation")
st.write("Translate between English and Hindi in real-time.")


if 'language' not in st.session_state:
    st.session_state.language = 'en_to_hi'
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""

def switch_language():
    if st.session_state.language == 'en_to_hi':
        st.session_state.language = 'hi_to_en'
    else:
        st.session_state.language = 'en_to_hi'

st.button("Switch Language", on_click=switch_language)


if st.session_state.language == 'en_to_hi':
    st.write("Current Language: English to Hindi")
else:
    st.write("Current Language: Hindi to English")


col1, col2 = st.columns(2)


with col1:
    input_text_area = st.empty()
with col2:
    translated_text_area = st.empty()


def translate(text, model, tokenizer):
    inputs = tokenizer.encode(text, return_tensors="pt", padding=True)
    outputs = model.generate(inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text


recognizer = sr.Recognizer()
mic = sr.Microphone()


def start_translating():
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        st.write("🎤 Listening...")
        while True:
            try:
                audio = recognizer.listen(source, timeout=1)
                speech_text = recognizer.recognize_google(audio, language='hi-IN' if st.session_state.language == 'hi_to_en' else 'en-US')

                st.session_state.input_text += " " + speech_text if st.session_state.input_text else speech_text
                
                if st.session_state.language == 'en_to_hi':
                    st.session_state.translated_text = translate(st.session_state.input_text, en_to_hi_model, en_to_hi_tokenizer)
                else:
                    st.session_state.translated_text = translate(st.session_state.input_text, hi_to_en_model, hi_to_en_tokenizer)
                
                input_text_area.text_area("Input Text", st.session_state.input_text, height=200, key="input_text_display")
                translated_text_area.text_area("Translated Text", st.session_state.translated_text, height=200, key="translated_text_display")
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                st.error(f"Could not request results; {e}")
            time.sleep(1)

if st.button("Start Translating"):
    start_translating()

if st.button("Clear"):
    st.session_state.input_text = ""
    st.session_state.translated_text = ""

input_text_area.text_area("Input Text", st.session_state.input_text, height=200, key="input_text_display_final")
translated_text_area.text_area("Translated Text", st.session_state.translated_text, height=200, key="translated_text_display_final")
