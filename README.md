
# 📽️ One Click Video Breakdown - Multimodal AI Pipeline

> **Transcribe, summarize, translate, and chat with any video — in one seamless AI-powered interface.**

---

## 🔍 Overview

This interactive web application enables users to upload or link to a video and automatically:

- 🎙️ Transcribe speech using Whisper
- ✍️ Generate a summary with GPT-4
- 🌍 Translate content into any language
- 🤖 Ask questions — and get answers based strictly on the video

It integrates cutting-edge AI models using **Whisper**, **GPT-4**, **LangChain**, and **Gradio**, forming a complete **multimodal pipeline**.

---

## 🎯 Motivation

In an era of long-form video content — podcasts, interviews, lectures, and tutorials — users often struggle to extract key insights without watching the full video.  
This project was built to:

- ⏳ **Save time** by automatically transcribing and summarizing video content.
- 🌍 **Break language barriers** by translating content into any language.
- 💬 **Enable interactive exploration** through a chatbot that answers specific questions based strictly on video content.
- ♿ **Support accessibility** for Deaf and hard-of-hearing users by providing high-quality, readable transcripts and summaries of spoken content.

---

## 🚀 Key Features

### 🔄 Multimodal Pipeline
- 🎧 **Speech-to-Text**: OpenAI Whisper for transcription
- 🧠 **Summarization**: GPT-4 condenses the transcript
- 🌐 **Translation**: Translate to any language, powered by GPT
- 🔊 **Text-to-Speech**: Convert translated text into natural audio using OpenAI TTS (Nova, Shimmer, Echo voices)

### 🧠 AI-Powered Q&A
- 💬 Ask any question about the video
- ✅ Answers are **fact-checked** using semantic vector search
- 🛡️ Non-relevant questions are rejected with: "This topic isn't covered in the video."
- 🌍 Arabic support: questions are translated internally to English for processing

### 🔎 Vector Search & Memory
- Transcripts are chunked and embedded via OpenAI Embeddings
- ChromaDB powers fast and relevant context retrieval
- Agent memory is preserved using LangChain's buffer

### 🖥️ Interactive Interface (Gradio)
- 📥 Upload videos or paste YouTube URLs
- 📄 View and hide transcripts dynamically
- 🌍 Choose your translation language and voice style
- 🤖 Chat in a dedicated tab with real-time response

---

## 🧠 Tech Stack

| Technology           | Role                                                              |
|----------------------|-------------------------------------------------------------------|
| **Whisper**          | Speech-to-text transcription                                      |
| **OpenAI GPT-4**     | Summarization, translation, and question answering                |
| **OpenAI TTS**       | Audio playback of translated text                                 |
| **LangChain**        | Agent framework, vector search, memory                            |
| **langchain-openai** | Updated OpenAI wrappers (ChatOpenAI & OpenAIEmbeddings)           |
| **ChromaDB**         | Vector storage and retrieval for semantic search                  |
| **Gradio**           | Web UI framework                                                  |
| **yt-dlp + ffmpeg**  | YouTube download & audio extraction                               |

---

## 📁 Code Structure

The entire application is implemented in **one Python file**, including:

- Model loading
- Transcription and summarization
- Vector DB creation
- Agent-based QA system
- Translation + TTS
- Full Gradio UI

---

## ⚙️ Setup Guide

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Set API Keys
```bash
export OPENAI_API_KEY=sk-...
export LANGCHAIN_API_KEY=...
```

### 4. Run the App
```bash
python your_script_name.py
```

---

## 🌐 Deployment Options

- 💻 **Localhost**: Automatically runs at `http://localhost:7860`
- ☁️ **Hugging Face Spaces** / **Streamlit Cloud** (requires light UI adaptation)
- 🔍 **LangSmith**: Trace LangChain agent interactions for debugging and optimization

---

## 🚧 Limitations

- Whisper accuracy may degrade with noisy audio
- GPT-generated answers depend on quality of transcript chunks
- Voice options limited to available OpenAI TTS models
- Agent strictly answers based only on the video transcript; off-topic questions are rejected with: "This topic isn't covered in the video."

---

## 💡 Future Enhancements

- ⏱️ Add clickable timestamps to summaries and chatbot answers
- 📄 Export transcripts and summaries to PDF/Word
- 😊 Add sentiment analysis to highlight emotional tone of the video
- 👤 Add user login for history/multi-session use
- 📊 Show source chunks or confidence in chatbot answers

---
##  Demo videos:
https://drive.google.com/drive/folders/1JGl3VrH6q8_mV9FxhsRfyZr7d1N4FPNu?usp=share_link

## 🙋 Contact

Built with ❤️ by **Raghad Alkhudair**  
Feel free to reach out with ideas, questions, or contributions.

---
