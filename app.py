import streamlit as st
from transformers import pipeline
import wikipedia
import PyPDF2
from io import BytesIO

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_model()

# -------------------- App Config --------------------
st.set_page_config(page_title="AI Text Summarizer", page_icon="📝", layout="centered")
st.title("📝 AI Text Summarizer")
st.write("Summarize text from **manual input**, **Wikipedia**, or **PDF files**.")

# -------------------- Create Tabs --------------------
tab1, tab2, tab3 = st.tabs(["✍ Manual Text", "🌐 Wikipedia", "📄 PDF"])

# -------- Tab 1: Manual Text Summarization --------
with tab1:
    st.subheader("✍ Manual Text Summarization")
    manual_text = st.text_area("Enter your text:", height=200, key="manual_text")
    if st.button("Summarize Text"):
        if manual_text.strip():
            with st.spinner("Summarizing your text..."):
                summary = summarizer(manual_text, max_length=100, min_length=30, do_sample=False)
            st.success(summary[0]['summary_text'])
        else:
            st.warning("Please enter some text.")

# -------- Tab 2: Wikipedia Summarization --------
with tab2:
    st.subheader("🌐 Wikipedia Summarization")
    wiki_topic = st.text_input("Enter Wikipedia topic:", key="wiki_topic")
    if st.button("Summarize Wikipedia"):
        if wiki_topic.strip():
            try:
                page = wikipedia.page(wiki_topic, auto_suggest=False)
                article = page.content
                limited_article = "\n".join(article.split("\n")[:3])
                with st.spinner("Fetching & summarizing from Wikipedia..."):
                    summary = summarizer(limited_article, max_length=150, min_length=50, do_sample=False)
                st.success(summary[0]['summary_text'])
                st.markdown(f"[Read Full Wikipedia Article]({page.url})", unsafe_allow_html=True)
            except wikipedia.exceptions.DisambiguationError as e:
                st.error(f"Topic is ambiguous. Suggested options: {e.options[:5]}")
            except wikipedia.exceptions.PageError:
                st.error("Page not found.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Enter a Wikipedia topic.")

# -------- Tab 3: PDF Summarization --------
with tab3:
    st.subheader("📄 PDF Summarization")
    pdf_file = st.file_uploader("Upload a PDF file:", type=["pdf"], key="pdf_input")
    if st.button("Summarize PDF"):
        if pdf_file:
            try:
                pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                with st.spinner("Summarizing from PDF..."):
                    summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
                st.success(summary[0]['summary_text'])
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
        else:
            st.warning("Please upload a PDF file.")

st.caption("⚡ Powered by Hugging Face Transformers, Wikipedia API, and PyPDF2")
