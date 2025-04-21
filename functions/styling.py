
import streamlit as st
import config as c


def page_config():
    
    st.set_page_config(
        page_title = c.page_title,
        layout="centered",
        page_icon=":material/neurology:",
        initial_sidebar_state="auto")
