
# External imports
import streamlit as st

# Python imports
import hmac
import os
import pathlib

# Local imports
import config as c
from functions.styling import page_config
from functions.menu import menu


### CSS AND STYLING

st.logo("images/logo_main.png", icon_image = "images/logo_small.png")

page_config()

def load_css(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = pathlib.Path("assets/styles.css")
load_css(css_path)

# Check if language is already in session_state, else initialize it with a default value
if 'language' not in st.session_state:
    st.session_state['language'] = "Svenska"  # Default language

### PASSWORD

if c.deployment == "streamlit":
    st.session_state["pwd_on"] = st.secrets.pwd_on
else:
    st.session_state["pwd_on"] = environ.get("pwd_on")

if st.session_state["pwd_on"] == "true":

    def check_password():

        if c.deployment == "streamlit":
            passwd1 = st.secrets["password"]
            passwd2 = st.secrets["password2"]
            passwd3 = st.secrets["password3"]
        else:
            passwd1 = environ.get("password")
            passwd2 = environ.get("password2")
            passwd3 = environ.get("password3")

        def password_entered():
            # Check if entered password matches either of the valid passwords
            if (hmac.compare_digest(st.session_state["password"], passwd1) or 
                hmac.compare_digest(st.session_state["password"], passwd2) or 
                hmac.compare_digest(st.session_state["password"], passwd3)):
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # Don't store the password.
            else:
                st.session_state["password_correct"] = False

        if st.session_state.get("password_correct", False):
            return True

        st.text_input("Lösenord", type="password", on_change=password_entered, key="password")
        if "password_correct" in st.session_state:
            st.error("😕 Ooops. Fel lösenord.")
        return False


    if not check_password():
        st.stop()

############

st.session_state["app_version"] = c.app_version
st.session_state["update_date"] = c.update_date

### SIDEBAR

menu()

### MAIN PAGE


st.markdown("# AI-labbet")

st.image("images/header.jpg")
st.markdown("###### ")

st.markdown("""
            __Välkommen till vår labbyta för generativ AI__"""
)
st.markdown("""
            Här i verktygslådan hittar du verktyg för att labba med generativ AI.
            """)
    
st.markdown("# ")