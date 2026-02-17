Generative AI-based Wikipedia & Custom Text Summarization

The Text Summarizer is a Generative AI–based web application that automatically generates concise, meaningful summaries from long text inputs or Wikipedia articles.
Built using Streamlit and Hugging Face Transformers, the app leverages a lightweight yet powerful BART-based model fine-tuned for abstractive summarization.

Users can either:

Paste custom text, or

Enter a Wikipedia topic and instantly get a summarized version

--> Key Features

- Abstractive Text Summarization using GenAI

- Custom Text Input – Paste any article, blog, or paragraph

- Wikipedia Article Summarization (real-time content fetch)

- Transformer-based LLM (DistilBART)

- Fast & Lightweight Model

- User-Friendly Streamlit Web Interface

--> Tech Stack
Category	Tools
Language	Python
UI Framework	Streamlit
AI Framework	Hugging Face Transformers
Model	sshleifer/distilbart-cnn-12-6
GenAI Technique	Abstractive Summarization
Retrieval	Wikipedia API (RAG-style pipeline)

--> Project Structure
Text-Summarizer/
│
├── app.py
├── requirements.txt
├── README.md
├── docs/
├── docs2/
└── .gitignore

--> How to Run the Project
1️. Clone the Repository
git clone https://github.com/Rajsinha7/Text-Summarizer.git
cd Text-Summarizer

2️. Install Dependencies
pip install -r requirements.txt

3️. Run the Application
streamlit run app.py


--> Requirements

Python 3.8+

Internet connection (for Wikipedia fetch & model download)

--> How It Works

User enters custom text or a Wikipedia topic
For Wikipedia input, content is fetched dynamically
Text is preprocessed and chunked (if required)
Transformer model generates an abstractive summary
Summary is displayed in a clean Streamlit UI

--> Concepts Demonstrated

Generative AI (NLG)
Transformer-based language models
Abstractive text summarization

Retrieval-Augmented Generation (RAG-style pipeline)

Prompt & text preprocessing
