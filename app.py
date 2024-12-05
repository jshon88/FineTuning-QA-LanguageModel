import streamlit as st
from LM import (
    read_squad,
    preprocess_training_examples,
    preprocess_validation_examples,
    SquadDataset,
    QAinference,
    compute_metrics,
    load_model_and_tokenizer
)
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm.auto import tqdm
import json

st.set_page_config(
    page_title="QA Language Model Inference",
    layout="centered",
    initial_sidebar_state="auto",
)
st.title("Question Answering Language Model")

st.sidebar.header("Model Selection")

available_models = [
    "deepset/bert-base-cased-squad2",
    "distilbert-base-uncased",
]

selected_model = st.sidebar.selectbox(
    "Select a Model",
    options=available_models,
    index=0  # Default selection
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_selected_model(model_name):
    model, tokenizer = load_model_and_tokenizer(model_name)
    model.to(device)
    return model, tokenizer

# Load model and tokenizer
with st.spinner(f"Loading {selected_model}..."):
    model, tokenizer = load_selected_model(selected_model)
st.success(f"{selected_model} loaded successfully!")

# User Input
st.header("Ask a Question")

question = st.text_input("Enter your question here:")
context = st.text_area("Enter the context here:")

# Perform inference
if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not context.strip():
        st.warning("Please enter the context.")
    else:
        with st.spinner("Generating answer..."):
            answer = QAinference(
                model=model,
                tokenizer=tokenizer,
                question=question,
                context=context,
                device=device,
                usepipeline=False  # Set to True to use pipeline
            )
        st.success("Answer Generated!")
        if isinstance(answer, dict):
            # If using pipeline
            st.write(f"**Answer:** {answer['answer']}")
            st.write(f"**Score:** {answer['score']:.4f}")
        else:
            # If using direct inference
            st.write(f"**Answer:** {answer}")