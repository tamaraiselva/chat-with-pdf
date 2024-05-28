from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template
import os

os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

# creating custom template to guide llm model
custom_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# extracting text from pdf
def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# converting text to chunks
def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)
    chunks = text_splitter.split_text(raw_text)
    return chunks

# using all-MiniLm embeddings model and faiss to get vectorstore
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vectorstore = faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# generating conversation chain  
def get_conversationchain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 512})

    memory = ConversationBufferMemory(memory_key='chat_history',
                                      return_messages=True,
                                      output_key='answer')  # using conversation buffer memory to hold past information
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory)
    return conversation_chain

# generating response from user queries and displaying them accordingly
def handle_question(question):
    if st.session_state.conversation is None:
        st.error("Please process the documents first.")
        return

    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response["chat_history"]
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content,), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

# display the source node with similarity score
def display_source_node(node, source_length=1500):
    st.write(f"Node ID: {node.metadata.get('id', 'N/A')}")
    st.write(f"Similarity Score: {node.metadata.get('score', 'N/A')}")
    st.write(f"Source Text: {node.page_content[:source_length]}")

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    question = st.text_input("Ask a question from your document:")
    if question:
        handle_question(question)
    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your PDF here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if docs:
                with st.spinner("Processing"):
                    # get the pdf
                    raw_text = get_pdf_text(docs)

                    # get the text chunks
                    text_chunks = get_chunks(raw_text)

                    # create vectorstore
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversationchain(vectorstore)
                    st.success("Processing done!")

                    # Use vectorstore directly to find top 2 nodes
                    retrievals = vectorstore.similarity_search("Can you tell me about the Paged Optimizers?", k=2)
                    for n in retrievals:
                        display_source_node(n, source_length=1500)
            else:
                st.error("Please upload at least one PDF document.")

        elif st.button("Clear"):
            st.session_state.conversation = None
            st.session_state.chat_history = None
            st.success("Session cleared!")

if __name__ == '__main__':
    main()