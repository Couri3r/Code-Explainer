import streamlit as st
from ask_model import explain_code
from streamlit_ace import st_ace

content = st_ace(language="python", theme="monokai")

if st.button("Generate Comment"):
    explanation = explain_code(content)
    st.write("Here's your comment:")
    st.write(explanation)
