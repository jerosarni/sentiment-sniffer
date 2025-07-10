import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment-Sniffer")
st.title("🕵️‍♂️ Sentiment-Sniffer")

text = st.text_area("Write something:", height=150)

# Cache the model so it loads only once
@st.cache_resource
def get_analyser():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )

if st.button("Analyse") and text:
    result = get_analyser()(text)[0]
    st.markdown(
        f"**Prediction:** {result['label']} "
        f"(confidence {result['score']:.2%})"
    )
