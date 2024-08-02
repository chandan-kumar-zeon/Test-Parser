import streamlit as st
import os
from helper import process_docs, generate_response

st.title("Test Parser Performance")

st.markdown("### Upload Document:")

uploaded_file = st.file_uploader("Choose a file", type='.pdf')

if 'queries' not in st.session_state:
    st.session_state.queries = []
    st.session_state.responses1 = []
    st.session_state.responses2 = []

if uploaded_file:
    if not os.path.exists("./Tested_Docs"):
        os.makedirs("./Tested_Docs")
        
    doc_path = f"./Tested_Docs/{uploaded_file.name}"
    with open(doc_path, "wb") as f:
        f.write(uploaded_file.read())

    try:
        with st.spinner("Processing document..."):
            query_engine_llama, query_engine_paddle, images = process_docs(doc_path)
    except Exception as e:
        st.warning(e)


    st.markdown("### Extracted Images:")
    for img in images:
        st.image(img, use_column_width=True)

    query = st.text_input("Enter your Query:")

    if st.button("Ask"):
        response1, response2 = generate_response(query, query_engine_llama, query_engine_paddle)
        
        st.session_state.queries.append(f"Query: {query}")
        st.session_state.responses1.append(f"LLama-Parser: {response1}\n")
        st.session_state.responses2.append(f"Paddle-OCR  : {response2}\n")

    for i in range(len(st.session_state.queries)):
        st.success(st.session_state.queries[i])
        st.info(st.session_state.responses1[i])
        st.info(st.session_state.responses2[i])
