import streamlit as st

st.title("Hello World")

text = st.text_input("What's your name")

if text.strip() != "":
    st.write(f"Hello, {text.strip()}!")
