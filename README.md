# doc_analyzer

Project Title:
Document Analysis Tool using LLMs for Summarization, Text Analysis, and Question-Answering

Project Overview:
This project is designed to explore how Large Language Models (LLMs) can be applied to efficiently analyze, summarize, and interact with large documents. The application provides users with the ability to upload PDF/DOCX files, which are then processed using state-of-the-art models to deliver key insights. The following features have been implemented:

Summarization: Using BART (facebook/bart-large-cnn), the tool generates concise summaries of lengthy documents, allowing users to quickly understand the essence of the content without reading through the entire document.

Text Analysis: Analyzes the document for key metrics such as word count, sentence count, and average sentence length. This is visualized using bar charts for easy interpretation.

Question-Answering: Leveraging RoBERTa (deepset/roberta-base-squad2), users can ask questions based on the document's content, and the model provides context-aware answers.

The application is built with Streamlit, making it interactive, user-friendly, and accessible for non-technical users.

LLM Models Used:
BART (facebook/bart-large-cnn):

Purpose: Used for document summarization.
Why: BART is a powerful transformer-based model that excels at generating summaries by focusing on both context and sentence structure, ensuring clarity and conciseness.
RoBERTa (deepset/roberta-base-squad2):

Purpose: Used for answering questions based on the document’s content.
Why: RoBERTa, a variant of BERT, is highly effective for question-answering tasks due to its ability to understand context and provide precise responses.

Project Goals:
Understand and Apply LLMs: Gain deeper insights into how LLMs function and can be applied across various industries like content creation, automation, and data analysis.
Improve Accuracy: Focus on enhancing the accuracy of the models for real-world applications. This is a critical aspect of working with LLMs, as better accuracy leads to more reliable insights.
Skill Development: This project is a stepping stone in enhancing my skill set in NLP, AI, and Machine Learning.
Features:
Upload Documents: Supports PDF and DOCX files for analysis.

Dynamic Summarization:

Input the length of the summary (min and max length) using sliders.
Generate real-time summaries from the uploaded document or text.
Text Analysis:

Get metrics on word count, sentence count, and average sentence length.
Visualize these metrics in a bar chart.
Question-Answering:

Ask any context-based question related to the document.
Receive answers with confidence scores.
Code Integration:


import streamlit as st

def show_project_details():
    st.subheader("Project Overview")
    st.markdown("""
    **Document Analysis Tool using LLMs for Summarization, Text Analysis, and Question-Answering**
    
    This tool leverages state-of-the-art Large Language Models (LLMs) to summarize, analyze, and interact with documents effectively. 
    Users can upload PDF/DOCX files, which are processed to deliver key insights.
    
    **Features**:
    1. **Summarization** using **BART** for concise document summaries.
    2. **Text Analysis** to provide insights into word count, sentence count, and average sentence length.
    3. **Question-Answering** using **RoBERTa** to answer context-based questions from the document.
    
    **LLM Models**:
    - **BART (facebook/bart-large-cnn)**: Used for document summarization.
    - **RoBERTa (deepset/roberta-base-squad2)**: Used for context-aware question-answering.

    **Project Goals**:
    - Apply LLMs in real-world scenarios.
    - Improve model accuracy for better real-world application.
    - Enhance skills in NLP, AI, and Machine Learning.
    """)
Instructions:
Call the show_project_details() function in your main code block to display this overview when the app runs.

You can add this section at the top of your app or as an additional sidebar feature for users to read while using the app.

Here’s how you can integrate it into your main code:

def main():
    st.title("Comprehensive Document Analyzer")

    # Add project details section
    st.sidebar.subheader("About the Project")
    show_project_details()

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
        # Add summarization, text analysis, and QA sections
        ...
        
if __name__ == "__main__":
    main()

Conclusion:
By integrating this detailed project description within your Streamlit app, you provide users with an overview of the project’s goals, the LLM models you’re using, and why they’ve been chosen. This will help others understand the technical aspects and the larger context of the project.
