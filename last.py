
"""
One Click Video Breakdown
--------------------------
This script builds a multimodal AI app using Gradio, Whisper, GPT-4, and LangChain.
It enables users to:
- Upload or link to a video
- Transcribe the audio
- Generate summaries and translations
- Interact with the video using a chatbot

Author: [Raghad]
"""

###############################
### 1. IMPORT DEPENDENCIES ###
###############################

# Core Python Libraries
import os                   # For file system operations
import tempfile             # For creating temporary files/directories

# Third-Party Libraries
import gradio as gr         # For building web UI
import whisper              # OpenAI's speech-to-text model
import yt_dlp               # YouTube video downloader
import ffmpeg               # Audio/video processing
from langdetect import detect  # Language detection

# LangChain Components
from langchain.schema import Document  # Data structure for text chunks
from langchain_community.vectorstores import Chroma  # Vector database
from langchain_community.chat_models import ChatOpenAI  # GPT-4 interface
from langchain_community.embeddings import OpenAIEmbeddings  # Text embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Text chunking
from langchain.agents import initialize_agent, AgentType  # AI agent setup
from langchain.tools import Tool  # Agent tools
from langchain.memory import ConversationBufferMemory  # Chat history storage

# OpenAI Integration
from openai import OpenAI  # Official OpenAI client
import openai              # Legacy OpenAI client (for TTS)

################################
### 2. CONFIGURATION SETUP ###
################################

# Set environment variables to avoid library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# API Keys 
openai_api_key = ""  
client = OpenAI(api_key=openai_api_key)

# LangChain Configuration
os.environ["LANGCHAIN_API_KEY"] = ""  
os.environ["LANGCHAIN_TRACING_V2"] = "true"       # Enable call tracing
os.environ["LANGCHAIN_PROJECT"] = "VideoQA-Agent" # Project name for tracking
openai.api_key = openai_api_key                   # Legacy OpenAI setup

################################
### 3. GLOBAL VARIABLES ###
################################

# AI Models
whisper_model = whisper.load_model("small")  # Balanced speed/accuracy STT model

# Application State
vector_store = None   # Will store video transcript embeddings
agent = None          # LangChain QA agent instance
full_transcript = ""  # Complete video transcription
chat_history = []     # Stores conversation history for UI

#####################################
### 4. LANGUAGE PROCESSING UTILS ###
#####################################

def is_arabic(text):
    """
    Detects if text is Arabic using langdetect
    Args:
        text: Input string to analyze
    Returns:
        bool: True if Arabic, False otherwise
    """
    try:
        return detect(text) == "ar"
    except:
        return False  # Fallback if detection fails

def translate_query_to_english(text):
    """
    Translates non-English text to English using GPT-4
    Args:
        text: Input text to translate
    Returns:
        str: English translation
    """
    prompt = f"Translate this to English:\n{text}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # Minimize creativity for accurate translation
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

################################
### 5. VIDEO PROCESSING ###
################################

def summarize_transcript_in_chunks(transcript, chunk_size=1500):
    """
    Summarizes a long transcript by chunking it and calling GPT-4 on each part.
    Returns a combined summary.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(transcript)

    summaries = []
    for i, chunk in enumerate(chunks):
        prompt = f"Summarize the following transcript chunk into key points:\n\n{chunk}"
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=600
        )
        summaries.append(response.choices[0].message.content.strip())

    combined = "\n\n".join(summaries)
    final_prompt = f"Summarize these points into one concise summary:\n\n{combined}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.4,
        max_tokens=600
    )
    return response.choices[0].message.content.strip()

def process_video(url_input, file_input):
    """
    Processes video from URL or file upload:
    1. Downloads/extracts audio
    2. Generates transcript
    3. Creates vector database
    4. Produces summary
    
    Args:
        url_input: YouTube URL
        file_input: Uploaded video file
    
    Returns:
        tuple: (full_transcript, summary)
    """
    global full_transcript, vector_store
    
    # Step 1: Input Handling
    if url_input:
        # YouTube download logic
        tmp = tempfile.mkdtemp()
        ydl_opts = {'format': 'best', 'outtmpl': os.path.join(tmp, "video.%(ext)s"), 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url_input, download=True)
            path = ydl.prepare_filename(info)
    elif file_input:
        # Local file handling
        path = file_input.name
    else:
        return gr.update(value="No input provided."), gr.update(value="")

    # Step 2: Audio Extraction
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    ffmpeg.input(path).output(audio_path, format='mp3', acodec='libmp3lame').run(overwrite_output=True)

    # Step 3: Transcription
    result = whisper_model.transcribe(audio_path)
    full_transcript = " ".join(segment['text'] for segment in result['segments'])

    # Step 4: Text Processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # Optimal for GPT context windows
        chunk_overlap=100    # Maintains context between chunks
    )
    documents = text_splitter.create_documents([full_transcript])
    
    # Step 5: Vector Storage
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = Chroma.from_documents(
        documents, 
        embeddings, 
        collection_name="video_chunks"  # Namespace for this video
    )
    return full_transcript, summary

summary = summarize_transcript_in_chunks(full_transcript)


#####################################
### 6. QUESTION ANSWERING SYSTEM ###
#####################################

def search_video_tool(query):
    """
    Semantic search tool for video content:
    1. Handles language translation
    2. Searches vector database
    3. Generates context-aware answers
    
    Args:
        query: User's question
    
    Returns:
        str: Answer or fallback message
    """
    if vector_store is None:
        return "❌ No transcript data available to search."

    # Language Handling
    query_en = translate_query_to_english(query) if is_arabic(query) else query
    
    # Vector Search
    results = vector_store.similarity_search(query_en, k=5)  # Top 5 chunks
    context_chunks = [doc.page_content for doc in results if doc.page_content.strip() != ""]
    context = "\n\n".join(context_chunks)

    # Fallback for no results
    if not context_chunks or len(context) < 50:
        return "This topic isn't covered in the video."

    # Answer Generation Prompt
    prompt = f"""
Based strictly on the content below, answer the question.
Do NOT assume or invent any information. If unclear, respond:
"This topic isn't covered in the video."

Video Content:
{context}

Question:
{query_en}
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # Low temperature for factual accuracy
        max_tokens=800
    )
    return response.choices[0].message.content.strip()

def build_agent():
    """
    Configures the LangChain agent with:
    - QA tool
    - GPT-4 backbone
    - Conversation memory
    """
    tools = [
        Tool(
            name="VideoQA",
            func=search_video_tool,
            description="ONLY answers questions about the video content."
        )
    ]
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0,          # Maximize determinism
        openai_api_key=openai_api_key
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True    # Store full message objects
    )
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,           # Enable logging
        memory=memory,
        handle_parsing_errors=True  # Graceful error handling
    )

def answer_with_agent(q):
    """
    Handles user questions by:
    1. Checking video context
    2. Routing to agent
    3. Updating chat history
    
    Args:
        q: User question
    
    Returns:
        list: Updated chat history
    """
    global agent, chat_history
    
    # Lazy agent initialization
    if agent is None:
        agent = build_agent()
    
    # Context check
    context = search_video_tool(q)
    if not context or len(context) < 20:
        reply = "هذا الموضوع غير مذكور في محتوى الفيديو." if is_arabic(q) else "This topic isn't covered in the video."
        chat_history.append({"role": "user", "content": q})
        chat_history.append({"role": "assistant", "content": reply})
        return chat_history
    
    # Generate answer
    prompt = f"""Answer based STRICTLY on this video content:
{context}

Question: {q}"""
    answer = agent.run(prompt)
    
    # Update history
    chat_history.append({"role": "user", "content": q})
    chat_history.append({"role": "assistant", "content": answer})
    return chat_history

    
# ################################
### 7. TRANSLATION & TTS ###
################################

def complete_translation_from_text(text, lang):
    """
    Translates text to target language using GPT-4
    
    Args:
        text: Source text
        lang: Target language code (e.g., "fr", "es")
    
    Returns:
        str: Translated text
    """
    prompt = f"Translate to {lang}:\n{text}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,  # Mild creativity for natural translations
        max_tokens=2000
    )
    return response.choices[0].message.content.strip()

def generate_openai_tts(text, voice_id, api_key):
    """
    Converts text to speech using OpenAI's TTS
    
    Args:
        text: Input text
        voice_id: Voice selection ("nova", "shimmer", etc.)
        api_key: OpenAI API key
    
    Returns:
        str: Path to generated audio file
    """
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    response = openai.audio.speech.create(
        model="tts-1",
        voice=voice_id,
        input=text
    )
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path

##########################
### 8. GRADIO UI ###
##########################

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎬 One Click Video Breakdown
    An AI-powered tool to analyze, summarize, translate, and interact with any video — effortlessly.
    """)

    with gr.Tab("📥 Upload video"):
        with gr.Row():
            url_input = gr.Textbox(label=" Paste URL")
            file_input = gr.File(label="Upload Video ", file_types=[".mp4", ".mov"])
        submit = gr.Button("🚀 Process Video")

        with gr.Row():
            show_btn = gr.Button("📄 Show Transcript")
            hide_btn = gr.Button("🔽 Hide Transcript", visible=False)
        transcript_box = gr.Textbox(label="Full Transcript", visible=False, lines=8)
        summary_box = gr.Textbox(label="Summary", lines=5)

    with gr.Tab("🌍 Translation"):
        section_text = gr.Textbox(label="Paste your text to translate", lines=5)
        with gr.Row():
            lang_input = gr.Textbox(label="Language")
            voice_dropdown = gr.Dropdown([
                ("Neutral (Nova)", "nova"),
                ("Female (Shimmer)", "shimmer"),
                ("Male (Echo)", "echo")
            ], value="nova", label="Select Voice Style")
        translate_btn = gr.Button("🌐 Translate and Listen")
        translated = gr.Textbox(label="Translation")
        audio_out = gr.Audio(label="Audio", type="filepath")

    with gr.Tab("🤖 Chat"):
        gr.Markdown("### Chat with your video")
        chatbot = gr.Chatbot(height=400)
        with gr.Row():
            question = gr.Textbox(label="Ask about the video", placeholder="Type your question...")
            clear_btn = gr.Button("🗑️ Clear", size="sm")
        submit_btn = gr.Button("Send", variant="primary")

        def respond(message, chat_history):
            """Handle chatbot input, get agent response, and update UI."""
            answer = answer_with_agent(message)
            chat_history.append((message, answer[-1]["content"]))
            return "", chat_history

        submit_btn.click(
            fn=respond,
            inputs=[question, chatbot],
            outputs=[question, chatbot]
        )

        clear_btn.click(lambda: None, None, chatbot, queue=False)

    submit.click(
        fn=process_video,
        inputs=[url_input, file_input],
        outputs=[transcript_box, summary_box]
    )

    show_btn.click(
        lambda: [gr.Textbox(visible=True), gr.Button(visible=True)],
        outputs=[transcript_box, hide_btn]
    )

    hide_btn.click(
        lambda: [gr.Textbox(visible=False), gr.Button(visible=False)],
        outputs=[transcript_box, hide_btn]
    )

    translate_btn.click(
        lambda text, lang, voice: (
            complete_translation_from_text(text, lang),
            generate_openai_tts(complete_translation_from_text(text, lang), voice, openai_api_key)
        ),
        inputs=[section_text, lang_input, voice_dropdown],
        outputs=[translated, audio_out]
    )

# Launch the app
demo.launch()