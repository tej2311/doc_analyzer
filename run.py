import streamlit as st
import PyPDF2
import docx
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
import time
import nltk

# Download necessary NLTK data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

download_nltk_data()

# Load LLM models for summarization and question-answering
@st.cache_resource
def load_models():
    summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    summarizer = pipeline("summarization", model=summarizer_model, tokenizer=summarizer_tokenizer)

    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return summarizer, qa_model

summarizer, qa_model = load_models()

# Extract text from uploaded PDF files
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract text from uploaded DOCX files
def read_docx(file):
    doc = docx.Document(file)
    return " ".join([para.text for para in doc.paragraphs])

# Summarize text using the LLM
def summarize_text(text, max_length=150, min_length=50):
    chunk_size = 512
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            st.warning(f"Error processing chunk: {str(e)}")
            continue

    combined_summary = " ".join(summaries)
    return combined_summary

# Answer question based on the provided document context
def answer_question(question, text):
    result = qa_model(question=question, context=text)
    return result['answer'], result['score']

# Analyze text for key statistics
def analyze_text(text):
    sentences = sent_tokenize(text)
    word_count = len(text.split())
    sentence_count = len(sentences)
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_sentence_length': word_count / sentence_count if sentence_count > 0 else 0
    }

# Visualize text statistics
def visualize_text_stats(stats):
    data = {
        'Metric': ['Word Count', 'Sentence Count', 'Avg. Sentence Length'],
        'Value': [stats['word_count'], stats['sentence_count'], stats['avg_sentence_length']]
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots()
    ax.bar(df['Metric'], df['Value'])
    ax.set_ylabel('Value')
    ax.set_title('Text Analysis Metrics')
    return fig

def main():
    st.title("Comprehensive Document Analyzer with Summarization and Q&A")

    uploaded_file = st.file_uploader("Upload a document (PDF or DOCX)", type=['pdf', 'docx'])
    
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            text = read_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = read_docx(uploaded_file)
        st.text_area("Extracted Text:", value=text[:1000] + "...", height=200)
    else:
        text = st.text_area("Or enter your document here:", height=200)

    if text:
        # Generate Summary Section
        st.subheader("Summarization")
        max_length = st.slider("Maximum summary length:", 50, 500, 150)
        min_length = st.slider("Minimum summary length:", 10, 100, 50)

        if st.button("Summarize"):
            with st.spinner("Generating summary..."):
                start_time = time.time()
                summary = summarize_text(text, max_length=max_length, min_length=min_length)
                end_time = time.time()
                st.write("**Summary:**")
                st.write(summary)
                st.write(f"Summarization took {end_time - start_time:.2f} seconds")
        
        # Text Analysis Section
        st.subheader("Text Analysis")
        if st.button("Analyze Text"):
            with st.spinner("Analyzing text..."):
                text_stats = analyze_text(text)
                st.write(f"Word Count: {text_stats['word_count']}")
                st.write(f"Sentence Count: {text_stats['sentence_count']}")
                st.write(f"Average Sentence Length: {text_stats['avg_sentence_length']:.2f} words")
                
                fig = visualize_text_stats(text_stats)
                st.pyplot(fig)

        # Question Answering Section
        st.subheader("Question Answering")
        question = st.text_input("Ask a question about the document:")
        if st.button("Answer Question"):
            if question:
                with st.spinner("Finding answer..."):
                    answer, confidence = answer_question(question, text)
                    st.write(f"**Answer:** {answer}")
                    st.write(f"**Confidence:** {confidence:.2f}")
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
