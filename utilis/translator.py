from deep_translator import GoogleTranslator
from deep_translator import exceptions as excp
import streamlit as st

def read_text(free_text_name,sidebar = True):
    if sidebar == True:
        text = st.sidebar.text_area(free_text_name,"")
    else:
        text = st.text_area(free_text_name,"")
    
    try:
        text = GoogleTranslator(source='auto', target='en').translate(text)
    except (excp.NotValidPayload, excp.NotValidLength) as e:
        text = ""
    return text
