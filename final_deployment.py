"""
One Click Video Breakdown
--------------------------
A multimodal AI application that processes video content through transcription, summarization,
translation, and interactive Q&A capabilities. Built with Gradio for the interface and leveraging
OpenAI's Whisper, GPT-4, and LangChain for AI processing.

Key Features:
- Video processing from URL or file upload
- Automatic transcription using Whisper
- Intelligent summarization with GPT-4
- Multilingual translation capabilities
- Interactive Q&A about video content
- Text-to-speech for translations

Author: Raghad
"""

###############################
### 1. IMPORT DEPENDENCIES ###
###############################

# Standard Library Imports
import os                   # File system operations
import tempfile             # Temporary file/directory management

# Third-Party Imports
import gradio as gr         # Web UI framework (v3.x)
import whisper              # OpenAI's speech-to-text model
import yt_dlp               # YouTube video downloading
import ffmpeg               # Audio/video processing
from langdetect import detect  # Language detection

# LangChain Components
from langchain.schema import Document  # Document representation
from langchain_community.vectorstores import Chroma  # Vector database
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Text chunking
from langchain.agents import initialize_agent, AgentType  # Agent framework
from langchain.tools import Tool, StructuredTool  # Agent tools
from langchain.memory import ConversationBufferMemory  # Conversation history
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # OpenAI integrations

# Pydantic for type validation
from pydantic import BaseModel, Field

# OpenAI Clients
from openai import OpenAI  # Official OpenAI client (v1.x+)
import openai              # Legacy OpenAI client (for TTS compatibility)

################################
### 2. CONFIGURATION SETUP ###
################################

# Environment Configuration
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Resolves library conflicts

# API Key Configuration
openai_api_key = ""  
client = OpenAI(api_key=openai_api_key)

# LangChain Tracking Configuration
os.environ["LANGCHAIN_API_KEY"] = ""  
os.environ["LANGCHAIN_TRACING_V2"] = "true"       # Enable execution tracing
os.environ["LANGCHAIN_PROJECT"] = "VideoQA-Agent" # Project name for tracking
openai.api_key = openai_api_key                   # Legacy OpenAI setup

################################
### 3. GLOBAL STATE ###
################################

# Model Initialization
whisper_model = whisper.load_model("small")  # Balanced STT model (speed/accuracy)

# Application State
vector_store = None   # Stores transcript embeddings for similarity search
agent = None          # LangChain QA agent instance
full_transcript = ""  # Complete video transcription text
chat_history = []     # Conversation history for UI display

#####################################
### 4. LANGUAGE PROCESSING UTILS ###
#####################################

def is_arabic(text: str) -> bool:
    """
    Determines if the input text is Arabic using language detection.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        bool: True if text is identified as Arabic, False otherwise
        
    Example:
        >>> is_arabic("مرحبا بالعالم")
        True
    """
    try:
        return detect(text) == "ar"
    except:
        return False  # Fallback for detection failures

def translate_query_to_english(text: str) -> str:
    """
    Translates non-English text to English using GPT-4.
    
    Args:
        text (str): Text to translate (typically Arabic in this context)
        
    Returns:
        str: English translation
        
    Note:
        Uses a temperature of 0 for maximum translation accuracy
    """
    prompt = f"Translate this to English:\n{text}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def summarize_transcript_in_chunks(transcript: str, chunk_size: int = 1500) -> str:
    """
    Generates a comprehensive summary of a long transcript by processing it in chunks.
    
    Args:
        transcript (str): Full video transcript text
        chunk_size (int): Maximum tokens per processing chunk
        
    Returns:
        str: Cohesive summary of the entire transcript
        
    Process:
        1. Splits transcript into manageable chunks
        2. Summarizes each chunk individually
        3. Combines chunk summaries into final comprehensive summary
    """
    # Initialize text splitter with overlap for context preservation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200
    )
    
    # Split transcript into chunks
    chunks = text_splitter.split_text(transcript)
    
    # Generate individual chunk summaries
    summaries = []
    for chunk in chunks:
        prompt = f"""Summarize the following transcript segment into clear bullet points:
        Focus on key ideas, events, and conclusions:
        \n\n{chunk}"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,  # Balanced creativity/factuality
            max_tokens=600
        )
        summaries.append(response.choices[0].message.content.strip())

    # Combine and refine all chunk summaries
    combined_summary = "\n".join(summaries)
    final_prompt = f"""This contains summarized bullet points from a video transcript.
    Synthesize these into a cohesive, well-structured summary covering all key points:
    \n\n{combined_summary}"""

    final_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.5,  # Slightly higher for narrative flow
        max_tokens=800
    )

    return final_response.choices[0].message.content.strip()

#######################################
### 5. VIDEO PROCESSING FUNCTIONS ###
#######################################

def process_video(url_input: str, file_input: str) -> tuple[str, str]:
    """
    Processes a video from URL or file upload to generate transcript and summary.
    
    Args:
        url_input (str): YouTube or other video URL
        file_input (str): Path to uploaded video file
        
    Returns:
        tuple: (full_transcript, summary)
        
    Workflow:
        1. Downloads video from URL or uses uploaded file
        2. Extracts audio using ffmpeg
        3. Transcribes audio with Whisper
        4. Generates embeddings for Q&A system
        5. Creates comprehensive summary
    """
    global full_transcript, vector_store

    # Handle video input source
    if url_input:
        # Download YouTube video
        tmp = tempfile.mkdtemp()
        ydl_opts = {
            'format': 'best',
            'outtmpl': os.path.join(tmp, "video.%(ext)s"),
            'quiet': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url_input, download=True)
            path = ydl.prepare_filename(info)
    elif file_input:
        path = file_input.name
    else:
        return "No input provided.", ""

    # Extract audio from video
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    ffmpeg.input(path).output(audio_path, format='mp3', acodec='libmp3lame').run(overwrite_output=True)

    # Transcribe audio to text
    result = whisper_model.transcribe(audio_path)
    full_transcript = " ".join(segment['text'] for segment in result['segments'])

    # Prepare text for vector storage
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = text_splitter.create_documents([full_transcript])

    # Create vector store for semantic search
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = Chroma.from_documents(
        documents,
        embeddings,
        collection_name="video_chunks"
    )

    # Generate and store summary
    summary = summarize_transcript_in_chunks(full_transcript)
    vector_store.add_documents([Document(page_content=summary)])
    
    return full_transcript, summary

####################################
### 6. Q&A AGENT IMPLEMENTATION ###
####################################

class VideoQASchema(BaseModel):
    """
    Pydantic model for validating Q&A tool inputs.
    
    Attributes:
        query (str): Question about the video content
    """
    query: str = Field(..., description="Question related to the uploaded video transcript")

def search_video_tool_func(query: str) -> str:
    """
    Core Q&A function that searches transcript and generates answers.
    
    Args:
        query (str): User's question about the video
        
    Returns:
        str: Answer derived from video content or fallback message
        
    Process:
        1. Translates non-English queries
        2. Performs semantic search on transcript
        3. Uses GPT-4 to generate context-aware answer
    """
    if vector_store is None:
        return "❌ No transcript data available to search."

    # Handle multilingual queries
    query_en = translate_query_to_english(query) if is_arabic(query) else query
    
    # Retrieve relevant transcript segments
    results = vector_store.similarity_search(query_en, k=5)
    context_chunks = [doc.page_content for doc in results if doc.page_content.strip()]
    context = "\n\n".join(context_chunks)[:12000]  # Limit context window

    # Handle empty results
    if not context or len(context) < 50:
        return "This topic isn't covered in the video."

    # Generate answer using context
    prompt = f"""
    Answer the question strictly using the video content below.
    If the answer cannot be found, respond: "This topic isn't covered in the video."
    
    Video Content:
    {context}
    
    Question:
    {query_en}
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# Initialize structured tool for the agent
search_video_tool = StructuredTool.from_function(
    func=search_video_tool_func,
    name="VideoQA",
    description="""Answer questions based on the video's transcript. 
    Use for ANY video-related questions including summaries. 
    If answer is unavailable, return: "This topic isn't covered in the video.\"""",
    args_schema=VideoQASchema,
    return_direct=True  # Bypass agent formatting for direct output
)

def build_agent() -> object:
    """
    Configures and initializes the LangChain Q&A agent.
    
    Returns:
        AgentExecutor: Configured agent instance
        
    Components:
        - GPT-4 as LLM backbone
        - Conversation memory
        - Constrained system prompt
        - VideoQA tool as sole capability
    """
    tools = [search_video_tool]
    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # System message constrains agent behavior
    system_message = """You are an AI assistant that ONLY answers questions about the uploaded video.
    ALWAYS use the VideoQA tool to answer questions.
    If the answer isn't in the video, respond: 'This topic isn't covered in the video.'
    NEVER use general knowledge - ONLY use VideoQA tool information."""
    
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={"system_message": system_message},
        verbose=True,
        memory=memory,
        handle_parsing_errors=True
    )

def answer_with_agent(question: str) -> list:
    """
    Handles user questions through the Q&A agent.
    
    Args:
        question (str): User's query
        
    Returns:
        list: Updated chat history with question/answer pair
        
    Side Effects:
        - Initializes agent if not exists
        - Updates global chat_history
    """
    global agent, chat_history

    if agent is None:
        agent = build_agent()

    try:
        answer = agent.run(question)
    except Exception as e:
        answer = f"⚠️ Error: {str(e)}"

    chat_history.append((question, answer))
    return chat_history

################################
### 7. TRANSLATION & TTS ###
################################

def complete_translation_from_text(text: str, lang: str) -> str:
    """
    Translates text to target language using GPT-4.
    
    Args:
        text (str): Source text to translate
        lang (str): Target language code (e.g., "fr", "es")
        
    Returns:
        str: Translated text
        
    Note:
        Uses temperature=0.4 for natural yet accurate translations
    """
    prompt = f"Translate to {lang}:\n{text}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=2000
    )
    return response.choices[0].message.content.strip()

def generate_openai_tts(text: str, voice_id: str, api_key: str) -> str:
    """
    Converts text to speech using OpenAI's TTS API.
    
    Args:
        text (str): Text to synthesize
        voice_id (str): Voice selection ("nova", "shimmer", etc.)
        api_key (str): OpenAI API key
        
    Returns:
        str: Path to generated audio file
        
    Note:
        Creates temporary MP3 file that persists until manually deleted
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
    # Application Header
    gr.Markdown("""
    # 🎬 One Click Video Breakdown
    An AI-powered tool to analyze, summarize, translate, and interact with any video — effortlessly.
    """)

    # Video Processing Tab
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

    # Translation Tab
    with gr.Tab("🌍 Translation"):
        section_text = gr.Textbox(label="Paste your text to translate", lines=5)
        with gr.Row():
            lang_input = gr.Textbox(label="Language")
            voice_dropdown = gr.Dropdown(
                choices=[
                    ("Neutral (Nova)", "nova"),
                    ("Female (Shimmer)", "shimmer"),
                    ("Male (Echo)", "echo")
                ],
                value="nova",
                label="Select Voice Style"
            )
        translate_btn = gr.Button("🌐 Translate and Listen")
        translated = gr.Textbox(label="Translation")
        audio_out = gr.Audio(label="Audio", type="filepath")

    # Chat Interface Tab
    with gr.Tab("🤖 Chat"):
        gr.Markdown("### Chat with your video")
        chatbot = gr.Chatbot(height=400)
        with gr.Row():
            question = gr.Textbox(
                label="Ask about the video",
                placeholder="Type your question...",
                container=False
            )
            clear_btn = gr.Button("🗑️ Clear", size="sm")
        submit_btn = gr.Button("Send", variant="primary")

        def respond(message: str, chat_history: list) -> tuple:
            """
            Handles chat interaction flow.
            
            Args:
                message (str): User's question
                chat_history (list): Current conversation history
                
            Returns:
                tuple: (empty_str, updated_chat_history)
            """
            answer = answer_with_agent(message) 
            if isinstance(answer, list) and len(answer) > 0 and isinstance(answer[-1], tuple):
                last_reply = answer[-1][1]
                chat_history.append((message, last_reply))
            else:
                chat_history.append((message, "❌ لم أستطع توليد رد."))

            return "", chat_history

        # Event Bindings
        submit_btn.click(
            fn=respond,
            inputs=[question, chatbot],
            outputs=[question, chatbot]
        )
        clear_btn.click(lambda: None, None, chatbot, queue=False)

    # Video Processing Bindings
    submit.click(
        fn=process_video,
        inputs=[url_input, file_input],
        outputs=[transcript_box, summary_box]
    )

    # Transcript Visibility Toggle
    show_btn.click(
        lambda: [gr.Textbox(visible=True), gr.Button(visible=True)],
        outputs=[transcript_box, hide_btn]
    )
    hide_btn.click(
        lambda: [gr.Textbox(visible=False), gr.Button(visible=False)],
        outputs=[transcript_box, hide_btn]
    )

    # Translation Binding
    translate_btn.click(
        lambda text, lang, voice: (
            complete_translation_from_text(text, lang),
            generate_openai_tts(complete_translation_from_text(text, lang), voice, openai_api_key)
        ),
        inputs=[section_text, lang_input, voice_dropdown],
        outputs=[translated, audio_out]
    )

# Launch the application
    demo.launch()