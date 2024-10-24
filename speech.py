import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import speech_recognition as sr
import time

# Load models and tokenizers
@st.cache_resource
def load_model_and_tokenizer(model_name):
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Model names
en_to_hi_model_name = 'Helsinki-NLP/opus-mt-en-hi'
hi_to_en_model_name = 'Helsinki-NLP/opus-mt-hi-en'

# Load models
en_to_hi_model, en_to_hi_tokenizer = load_model_and_tokenizer(en_to_hi_model_name)
hi_to_en_model, hi_to_en_tokenizer = load_model_and_tokenizer(hi_to_en_model_name)

# Initialize Streamlit app
st.title("Real-time Language Translation")
st.write("Translate between English and Hindi in real-time.")

# Language switch button
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

# Display current selected language
st.write("Current Language:", "English to Hindi" if st.session_state.language == 'en_to_hi' else "Hindi to English")

# Create columns for left-right layout
col1, col2 = st.columns(2)

# Input and output text areas
input_text_area = col1.text_area("Input Text", st.session_state.input_text, height=200, key="input_text_display")
translated_text_area = col2.text_area("Translated Text", st.session_state.translated_text, height=200, key="translated_text_display")

# Translation function
def translate(text, model, tokenizer):
    inputs = tokenizer.encode(text, return_tensors="pt", padding=True)
    outputs = model.generate(inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# Speech recognition setup
recognizer = sr.Recognizer()
mic = sr.Microphone()

# Real-time translation
def start_translating():
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        st.write("🎤 Listening...")
        while True:
            try:
                audio = recognizer.listen(source, timeout=1)
                speech_text = recognizer.recognize_google(audio, language='hi-IN' if st.session_state.language == 'hi_to_en' else 'en-US')
                
                # Append recognized text to the session state
                st.session_state.input_text += " " + speech_text if st.session_state.input_text else speech_text
                
                # Translate the text
                if st.session_state.language == 'en_to_hi':
                    st.session_state.translated_text = translate(st.session_state.input_text, en_to_hi_model, en_to_hi_tokenizer)
                else:
                    st.session_state.translated_text = translate(st.session_state.input_text, hi_to_en_model, hi_to_en_tokenizer)
                
                # Update the text areas
                st.session_state.input_text = input_text_area  # Update session state with new input text
                st.session_state.translated_text = translated_text_area  # Update session state with new translated text
                
                # Refresh text areas
                input_text_area.text_area("Input Text", st.session_state.input_text, height=200, key="input_text_display_final")
                translated_text_area.text_area("Translated Text", st.session_state.translated_text, height=200, key="translated_text_display_final")

            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                st.error(f"Could not request results; {e}")
            time.sleep(1)

# Start translation button
if st.button("Start Translating"):
    start_translating()

# Clear button to reset the recognized speech
if st.button("Clear"):
    st.session_state.input_text = ""
    st.session_state.translated_text = ""
    # Clear the displayed text areas
    input_text_area.text_area("Input Text", st.session_state.input_text, height=200, key="input_text_display_final")
    translated_text_area.text_area("Translated Text", st.session_state.translated_text, height=200, key="translated_text_display_final")
