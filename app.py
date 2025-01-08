import streamlit as st
import anthropic, openai, base64, cv2, glob, json, math, os, pytz, random, re, requests, textract, time, zipfile
import plotly.graph_objects as go
import streamlit.components.v1 as components
from datetime import datetime
from audio_recorder_streamlit import audio_recorder
from bs4 import BeautifulSoup
from collections import defaultdict, deque, Counter
from dotenv import load_dotenv
from gradio_client import Client
from huggingface_hub import InferenceClient
from io import BytesIO
from PIL import Image
from PyPDF2 import PdfReader
from urllib.parse import quote
from xml.etree import ElementTree as ET
from openai import OpenAI
import extra_streamlit_components as stx
from streamlit.runtime.scriptrunner import get_script_run_ctx
import asyncio
import edge_tts
from streamlit_marquee import streamlit_marquee

# 🎯 1. Core Configuration & Setup
st.set_page_config(
    page_title="🚲TalkingAIResearcher🏆",
    page_icon="🚲🏆",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://huggingface.co/awacke1',
        'Report a bug': 'https://huggingface.co/spaces/awacke1',
        'About': "🚲TalkingAIResearcher🏆"
    }
)
load_dotenv()

# Add available English voices for Edge TTS
EDGE_TTS_VOICES = [
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-AU-NatashaNeural",
    "en-AU-WilliamNeural",
    "en-CA-ClaraNeural",
    "en-CA-LiamNeural"
]

# Initialize session state variables
if 'marquee_settings' not in st.session_state:
    st.session_state['marquee_settings'] = {
        "background": "#1E1E1E",
        "color": "#FFFFFF",
        "font-size": "14px",
        "animationDuration": "10s",
        "width": "100%",
        "lineHeight": "35px"
    }

if 'tts_voice' not in st.session_state:
    st.session_state['tts_voice'] = EDGE_TTS_VOICES[0]
if 'audio_format' not in st.session_state:
    st.session_state['audio_format'] = 'mp3'
if 'transcript_history' not in st.session_state:
    st.session_state['transcript_history'] = []
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'openai_model' not in st.session_state:
    st.session_state['openai_model'] = "gpt-4o-2024-05-13"
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'last_voice_input' not in st.session_state:
    st.session_state['last_voice_input'] = ""
if 'editing_file' not in st.session_state:
    st.session_state['editing_file'] = None
if 'edit_new_name' not in st.session_state:
    st.session_state['edit_new_name'] = ""
if 'edit_new_content' not in st.session_state:
    st.session_state['edit_new_content'] = ""
if 'viewing_prefix' not in st.session_state:
    st.session_state['viewing_prefix'] = None
if 'should_rerun' not in st.session_state:
    st.session_state['should_rerun'] = False
if 'old_val' not in st.session_state:
    st.session_state['old_val'] = None
if 'last_query' not in st.session_state:
    st.session_state['last_query'] = ""
if 'marquee_content' not in st.session_state:
    st.session_state['marquee_content'] = "🚀 Welcome to TalkingAIResearcher | 🤖 Your Research Assistant"

# 🔑 2. API Setup & Clients
openai_api_key = os.getenv('OPENAI_API_KEY', "")
anthropic_key = os.getenv('ANTHROPIC_API_KEY_3', "")
xai_key = os.getenv('xai',"")
if 'OPENAI_API_KEY' in st.secrets:
    openai_api_key = st.secrets['OPENAI_API_KEY']
if 'ANTHROPIC_API_KEY' in st.secrets:
    anthropic_key = st.secrets["ANTHROPIC_API_KEY"]

openai.api_key = openai_api_key
claude_client = anthropic.Anthropic(api_key=anthropic_key)
openai_client = OpenAI(api_key=openai.api_key, organization=os.getenv('OPENAI_ORG_ID'))
HF_KEY = os.getenv('HF_KEY')
API_URL = os.getenv('API_URL')

# Constants
FILE_EMOJIS = {
    "md": "📝",
    "mp3": "🎵",
    "wav": "🔊"
}

def get_central_time():
    """Get current time in US Central timezone"""
    central = pytz.timezone('US/Central')
    return datetime.now(central)

def format_timestamp_prefix():
    """Generate timestamp prefix in format MM_dd_yy_hh_mm_AM/PM"""
    ct = get_central_time()
    return ct.strftime("%m_%d_%y_%I_%M_%p")

def initialize_marquee_settings():
    """Initialize marquee settings in session state"""
    if 'marquee_settings' not in st.session_state:
        st.session_state['marquee_settings'] = {
            "background": "#1E1E1E",
            "color": "#FFFFFF",
            "font-size": "14px",
            "animationDuration": "10s",
            "width": "100%",
            "lineHeight": "35px"
        }

def get_marquee_settings():
    """Get or update marquee settings from session state"""
    initialize_marquee_settings()
    return st.session_state['marquee_settings']

def update_marquee_settings_ui():
    """Update marquee settings via UI controls"""
    initialize_marquee_settings()
    st.sidebar.markdown("### 🎯 Marquee Settings")
    cols = st.sidebar.columns(2)
    with cols[0]:
        bg_color = st.color_picker("🎨 Background", 
                                  st.session_state['marquee_settings']["background"], 
                                  key="bg_color_picker")
        text_color = st.color_picker("✍️ Text", 
                                    st.session_state['marquee_settings']["color"], 
                                    key="text_color_picker")
    with cols[1]:
        font_size = st.slider("📏 Size", 10, 24, 14, key="font_size_slider")
        duration = st.slider("⏱️ Speed", 1, 20, 10, key="duration_slider")

    st.session_state['marquee_settings'].update({
        "background": bg_color,
        "color": text_color,
        "font-size": f"{font_size}px",
        "animationDuration": f"{duration}s"
    })

def display_marquee(text, settings, key_suffix=""):
    """Display marquee with given text and settings"""
    truncated_text = text[:280] + "..." if len(text) > 280 else text
    streamlit_marquee(
        content=truncated_text,
        **settings,
        key=f"marquee_{key_suffix}"
    )
    st.write("")

def get_high_info_terms(text: str, top_n=10) -> list:
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'])
    words = re.findall(r'\b\w+(?:-\w+)*\b', text.lower())
    bi_grams = [' '.join(pair) for pair in zip(words, words[1:])]
    combined = words + bi_grams
    filtered = [term for term in combined if term not in stop_words and len(term.split()) <= 2]
    counter = Counter(filtered)
    return [term for term, freq in counter.most_common(top_n)]

def clean_text_for_filename(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    words = text.split()
    stop_short = set(['the', 'and', 'for', 'with', 'this', 'that'])
    filtered = [w for w in words if len(w) > 3 and w not in stop_short]
    return '_'.join(filtered)[:200]

def generate_filename(prompt, response, file_type="md"):
    prefix = format_timestamp_prefix() + "_"
    combined = (prompt + " " + response).strip()
    info_terms = get_high_info_terms(combined, top_n=10)
    snippet = (prompt[:100] + " " + response[:100]).strip()
    snippet_cleaned = clean_text_for_filename(snippet)
    name_parts = info_terms + [snippet_cleaned]
    full_name = '_'.join(name_parts)
    if len(full_name) > 150:
        full_name = full_name[:150]
    return f"{prefix}{full_name}.{file_type}"

def create_file(prompt, response, file_type="md"):
    filename = generate_filename(prompt.strip(), response.strip(), file_type)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(prompt + "\n\n" + response)
    return filename

def get_download_link(file, file_type="zip"):
    with open(file, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    if file_type == "zip":
        return f'<a href="data:application/zip;base64,{b64}" download="{os.path.basename(file)}">📂 Download {os.path.basename(file)}</a>'
    elif file_type == "mp3":
        return f'<a href="data:audio/mpeg;base64,{b64}" download="{os.path.basename(file)}">🎵 Download {os.path.basename(file)}</a>'
    elif file_type == "wav":
        return f'<a href="data:audio/wav;base64,{b64}" download="{os.path.basename(file)}">🔊 Download {os.path.basename(file)}</a>'
    elif file_type == "md":
        return f'<a href="data:text/markdown;base64,{b64}" download="{os.path.basename(file)}">📝 Download {os.path.basename(file)}</a>'
    else:
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(file)}">Download {os.path.basename(file)}</a>'

def clean_for_speech(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.replace("</s>", " ")
    text = text.replace("#", "")
    text = re.sub(r"\(https?:\/\/[^\)]+\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

async def edge_tts_generate_audio(text, voice="en-US-AriaNeural", rate=0, pitch=0, file_format="mp3"):
    text = clean_for_speech(text)
    if not text.strip():
        return None
    rate_str = f"{rate:+d}%"
    pitch_str = f"{pitch:+d}Hz"
    communicate = edge_tts.Communicate(text, voice, rate=rate_str, pitch=pitch_str)
    out_fn = generate_filename(text, text, file_type=file_format)
    await communicate.save(out_fn)
    return out_fn

def speak_with_edge_tts(text, voice="en-US-AriaNeural", rate=0, pitch=0, file_format="mp3"):
    return asyncio.run(edge_tts_generate_audio(text, voice, rate, pitch, file_format))

def play_and_download_audio(file_path, file_type="mp3"):
    if file_path and os.path.exists(file_path):
        st.audio(file_path)
        dl_link = get_download_link(file_path, file_type=file_type)
        st.markdown(dl_link, unsafe_allow_html=True)

def save_qa_with_audio(question, answer, voice=None):
    """Save Q&A to markdown and generate audio"""
    if not voice:
        voice = st.session_state['tts_voice']
    
    # Create markdown file
    combined_text = f"# Question\n{question}\n\n# Answer\n{answer}"
    md_file = create_file(question, answer, "md")
    
    # Generate audio file
    audio_text = f"Question: {question}\n\nAnswer: {answer}"
    audio_file = speak_with_edge_tts(
        audio_text,
        voice=voice,
        file_format=st.session_state['audio_format']
    )
    
    return md_file, audio_file

def process_paper_content(paper):
    marquee_text = f"📄 {paper['title']} | 👤 {paper['authors'][:100]} | 📝 {paper['summary'][:100]}"
    audio_text = f"{paper['title']} by {paper['authors']}. {paper['summary']}"
    return marquee_text, audio_text

def create_paper_audio_files(papers, input_question):
    for paper in papers:
        try:
            marquee_text, audio_text = process_paper_content(paper)
            
            audio_text = clean_for_speech(audio_text)
            file_format = st.session_state['audio_format']
            audio_file = speak_with_edge_tts(audio_text, 
                                           voice=st.session_state['tts_voice'], 
                                           file_format=file_format)
            paper['full_audio'] = audio_file
            
            st.write(f"### {FILE_EMOJIS.get(file_format, '')} {os.path.basename(audio_file)}")
            play_and_download_audio(audio_file, file_type=file_format)
            paper['marquee_text'] = marquee_text
            
        except Exception as e:
            st.warning(f"Error processing paper {paper['title']}: {str(e)}")
            paper['full_audio'] = None
            paper['marquee_text'] = None

def display_papers(papers, marquee_settings):
    st.write("## Research Papers")
    
    papercount = 0
    for paper in papers:
        papercount += 1
        if papercount <= 20:
            if paper.get('marquee_text'):
                display_marquee(paper['marquee_text'], 
                              marquee_settings,
                              key_suffix=f"paper_{papercount}")
            
            with st.expander(f"{papercount}. 📄 {paper['title']}", expanded=True):
                st.markdown(f"**{paper['date']} | {paper['title']} | ⬇️**")
                st.markdown(f"*{paper['authors']}*")
                st.markdown(paper['summary'])
                
                if paper.get('full_audio'):
                    st.write("📚 Paper Audio")
                    file_ext = os.path.splitext(paper['full_audio'])[1].lower().strip('.')
                    if file_ext in ['mp3', 'wav']:
                        st.audio(paper['full_audio'])

def parse_arxiv_refs(ref_text: str):
    if not ref_text:
        return []

    results = []
    current_paper = {}
    lines = ref_text.split('\n')
    
    for i, line in enumerate(lines):
        if line.count('|') == 2:
            if current_paper:
                results.append(current_paper)
                if len(results) >= 20:
                    break
            
            try:
                header_parts = line.strip('* ').split('|')
                date = header_parts[0].strip()
                title = header_parts[1].strip()
                url_match = re.search(r'(https://arxiv.org/\S+)', line)
                url = url_match.group(1) if url_match else f"paper_{len(results)}"
                
                current_paper = {
                    'date': date,
                    'title': title,
                    'url': url,
                    'authors': '',
                    'summary': '',
                    'content_start': i + 1
                }
            except Exception as e:
                st.warning(f"Error parsing paper header: {str(e)}")
                current_paper = {}
                continue
        
        elif current_paper:
            if not current_paper['authors']:
                current_paper['authors'] = line.strip('* ')
            else:
                if current_paper['summary']:
                    current_paper['summary'] += ' ' + line.strip()
                else:
                    current_paper['summary'] = line.strip()
    
    if current_paper:
        results.append(current_paper)
    
    return results[:20]

def perform_ai_lookup(q, vocal_summary=True, extended_refs=False, 
                     titles_summary=True, full_audio=False):
    start = time.time()

    client = Client("awacke1/Arxiv-Paper-Search-And-QA-RAG-Pattern")
    refs = client.predict(q, 20, "Semantic Search", 
                         "mistralai/Mixtral-8x7B-Instruct-v0.1",
                         api_name="/update_with_rag_md")[0]
    r2 = client.predict(q, "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                       True, api_name="/ask_llm")

    result = f"### 🔎 {q}\n\n{r2}\n\n{refs}"
    st.markdown(result)

    md_file, audio_file = save_qa_with_audio(q, result)
    
    st.subheader("📝 Main Response Audio")
    play_and_download_audio(audio_file, st.session_state['audio_format'])

    papers = parse_arxiv_refs(refs)
    if papers:
        create_paper_audio_files(papers, input_question=q)
        display_papers(papers, get_marquee_settings())
    else:
        st.warning("No papers found in the response.")

    elapsed = time.time()-start
    st.write(f"**Total Elapsed:** {elapsed:.2f} s")
    return result

def process_voice_input(text):
    if not text:
        return
        
    st.subheader("🔍 Search Results")
    result = perform_ai_lookup(
        text, 
        vocal_summary=True,
        extended_refs=False,
        titles_summary=True,
        full_audio=True
    )
    
    md_file, audio_file = save_qa_with_audio(text, result)
    
    st.subheader("📝 Generated Files")
    st.write(f"Markdown: {md_file}")
    st.write(f"Audio: {audio_file}")
    play_and_download_audio(audio_file, st.session_state['audio_format'])

def load_files_for_sidebar():
    md_files = glob.glob("*.md")
    mp3_files = glob.glob("*.mp3")
    wav_files = glob.glob("*.wav")
    
    md_files = [f for f in md_files if os.path.basename(f).lower() != 'readme.md']
    all_files = md_files + mp3_files + wav_files

    groups = defaultdict(list)
    prefix_length = len("MM_dd_yy_hh_mm_AP")
    
    for f in all_files:
        basename = os.path.basename(f)
        if len(basename) >= prefix_length and '_' in basename:
            group_name = basename[:prefix_length]
            groups[group_name].append(f)
        else:
            groups['Other'].append(f)
            
    sorted_groups = sorted(groups.items(), 
                         key=lambda x: x[0] if x[0] != 'Other' else '', 
                         reverse=True)
    return sorted_groups

def display_file_manager_sidebar(groups_sorted):
    st.sidebar.title("🎵 Audio & Docs Manager")

    all_md = []
    all_mp3 = []
    all_wav = []
    for _, files in groups_sorted:
        for f in files:
            if f.endswith(".md"):
                all_md.append(f)
            elif f.endswith(".mp3"):
                all_mp3.append(f)
            elif f.endswith(".wav"):
                all_wav.append(f)

    col1, col2, col3, col4 = st.sidebar.columns(4)
    with col1:
        if st.button("🗑 DelMD"):
            for f in all_md:
                os.remove(f)
            st.session_state.should_rerun = True
    with col2:
        if st.button("🗑 DelMP3"):
            for f in all_mp3:
                os.remove(f)
            st.session_state.should_rerun = True
    with col3:
        if st.button("🗑 DelWAV"):
            for f in all_wav:
                os.remove(f)
            st.session_state.should_rerun = True
    with col4:
        if st.button("⬇️ ZipAll"):
            zip_name = create_zip_of_files(all_md, all_mp3, all_wav, st.session_state.get('last_query', ''))
            if zip_name:
                st.sidebar.markdown(get_download_link(zip_name, "zip"), unsafe_allow_html=True)

    for group_name, files in groups_sorted:
        if group_name == 'Other':
            group_label = 'Other Files'
        else:
            try:
                timestamp_dt = datetime.strptime(group_name, "%m_%d_%y_%I_%M_%p")
                group_label = timestamp_dt.strftime("%b %d, %Y %I:%M %p")
            except ValueError:
                group_label = group_name

        with st.sidebar.expander(f"📁 {group_label} ({len(files)})", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                if st.button("👀 View", key=f"view_group_{group_name}"):
                    st.session_state.viewing_prefix = group_name
            with c2:
                if st.button("🗑 Del", key=f"del_group_{group_name}"):
                    for f in files:
                        os.remove(f)
                    st.success(f"Deleted group {group_label}!")
                    st.session_state.should_rerun = True
                    
            for f in files:
                fname = os.path.basename(f)
                ext = os.path.splitext(fname)[1].lower()
                emoji = FILE_EMOJIS.get(ext.strip('.'), '')
                mtime = os.path.getmtime(f)
                ctime = datetime.fromtimestamp(mtime).strftime("%I:%M:%S %p")
                st.write(f"{emoji} **{fname}** - {ctime}")

def create_zip_of_files(md_files, mp3_files, wav_files, input_question):
    md_files = [f for f in md_files if os.path.basename(f).lower() != 'readme.md']
    all_files = md_files + mp3_files + wav_files
    if not all_files:
        return None

    all_content = []
    for f in all_files:
        if f.endswith('.md'):
            with open(f, 'r', encoding='utf-8') as file:
                all_content.append(file.read())
        elif f.endswith('.mp3') or f.endswith('.wav'):
            basename = os.path.splitext(os.path.basename(f))[0]
            words = basename.replace('_', ' ')
            all_content.append(words)
    
    all_content.append(input_question)
    combined_content = " ".join(all_content)
    info_terms = get_high_info_terms(combined_content, top_n=10)
    
    timestamp = format_timestamp_prefix()
    name_text = '_'.join(term.replace(' ', '-') for term in info_terms[:10])
    zip_name = f"{timestamp}_{name_text}.zip"
    
    with zipfile.ZipFile(zip_name, 'w') as z:
        for f in all_files:
            z.write(f)
    
    return zip_name

def main():
    # Update marquee settings UI first
    update_marquee_settings_ui()
    marquee_settings = get_marquee_settings()
    
    # Initial welcome marquee
    display_marquee(st.session_state['marquee_content'], 
                   {**marquee_settings, "font-size": "28px", "lineHeight": "50px"},
                   key_suffix="welcome")

    # Load files for sidebar
    groups_sorted = load_files_for_sidebar()
    
    # Update marquee content when viewing files
    if st.session_state.viewing_prefix:
        for group_name, files in groups_sorted:
            if group_name == st.session_state.viewing_prefix:
                for f in files:
                    if f.endswith('.md'):
                        with open(f, 'r', encoding='utf-8') as file:
                            st.session_state['marquee_content'] = file.read()[:280]

    # Voice Settings
    st.sidebar.markdown("### 🎤 Voice Settings")
    selected_voice = st.sidebar.selectbox(
        "Select TTS Voice:",
        options=EDGE_TTS_VOICES,
        index=EDGE_TTS_VOICES.index(st.session_state['tts_voice'])
    )
    
    # Audio Format Settings
    st.sidebar.markdown("### 🔊 Audio Format")
    selected_format = st.sidebar.radio(
        "Choose Audio Format:",
        options=["MP3", "WAV"],
        index=0
    )
    
    if selected_voice != st.session_state['tts_voice']:
        st.session_state['tts_voice'] = selected_voice
        st.rerun()
    if selected_format.lower() != st.session_state['audio_format']:
        st.session_state['audio_format'] = selected_format.lower()
        st.rerun()

    # Main Interface
    tab_main = st.radio("Action:", ["🎤 Voice", "📸 Media", "🔍 ArXiv", "📝 Editor"], 
                       horizontal=True)

    mycomponent = components.declare_component("mycomponent", path="mycomponent")
    val = mycomponent(my_input_value="Hello")

    if val:
        val_stripped = val.replace('\\n', ' ')
        edited_input = st.text_area("✏️ Edit Input:", value=val_stripped, height=100)
        
        run_option = st.selectbox("Model:", ["Arxiv"])
        col1, col2 = st.columns(2)
        with col1:
            autorun = st.checkbox("⚙ AutoRun", value=True)
        with col2:
            full_audio = st.checkbox("📚FullAudio", value=False)

        input_changed = (val != st.session_state.old_val)

        if autorun and input_changed:
            st.session_state.old_val = val
            st.session_state.last_query = edited_input
            result = perform_ai_lookup(edited_input, vocal_summary=True, extended_refs=False, 
                                    titles_summary=True, full_audio=full_audio)
        else:
            if st.button("▶ Run"):
                st.session_state.old_val = val
                st.session_state.last_query = edited_input
                result = perform_ai_lookup(edited_input, vocal_summary=True, extended_refs=False, 
                                        titles_summary=True, full_audio=full_audio)
    

    if tab_main == "🔍 ArXiv":
        st.subheader("🔍 Query ArXiv")
        q = st.text_input("🔍 Query:", key="arxiv_query")
    
        st.markdown("### 🎛 Options")
        vocal_summary = st.checkbox("🎙ShortAudio", value=True, key="option_vocal_summary")
        extended_refs = st.checkbox("📜LongRefs", value=False, key="option_extended_refs")
        titles_summary = st.checkbox("🔖TitlesOnly", value=True, key="option_titles_summary")
        full_audio = st.checkbox("📚FullAudio", value=False, key="option_full_audio")
        full_transcript = st.checkbox("🧾FullTranscript", value=False, key="option_full_transcript")
        
        
        if q and st.button("🔍Run"):
            st.session_state.last_query = q
            result = perform_ai_lookup(q, vocal_summary=vocal_summary, extended_refs=extended_refs, 
                                     titles_summary=titles_summary, full_audio=full_audio)
            if full_transcript:
                create_file(q, result, "md")

    elif tab_main == "🎤 Voice":
        st.subheader("🎤 Voice Input")
        user_text = st.text_area("💬 Message:", height=100)
        user_text = user_text.strip().replace('\n', ' ')

        if st.button("📨 Send"):
            process_voice_input(user_text)
            
        st.subheader("📜 Chat History")
        for c in st.session_state.chat_history:
            st.write("**You:**", c["user"])
            st.write("**Response:**", c["claude"])

    elif tab_main == "📸 Media":
        st.header("📸 Images & 🎥 Videos")
        tabs = st.tabs(["🖼 Images", "🎥 Video"])
        with tabs[0]:
            imgs = glob.glob("*.png") + glob.glob("*.jpg")
            if imgs:
                c = st.slider("Cols", 1, 5, 3)
                cols = st.columns(c)
                for i, f in enumerate(imgs):
                    with cols[i % c]:
                        st.image(Image.open(f), use_container_width=True)
                        if st.button(f"👀 Analyze {os.path.basename(f)}", key=f"analyze_{f}"):
                            response = openai_client.chat.completions.create(
                                model=st.session_state["openai_model"],
                                messages=[
                                    {"role": "system", "content": "Analyze the image content."},
                                    {"role": "user", "content": [
                                        {"type": "image_url", 
                                         "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(open(f, 'rb').read()).decode()}"}}
                                    ]}
                                ]
                            )
                            st.markdown(response.choices[0].message.content)
            else:
                st.write("No images found.")
        
        with tabs[1]:
            vids = glob.glob("*.mp4")
            if vids:
                for v in vids:
                    with st.expander(f"🎥 {os.path.basename(v)}"):
                        st.video(v)
                        if st.button(f"Analyze {os.path.basename(v)}", key=f"analyze_{v}"):
                            frames = process_video(v)
                            response = openai_client.chat.completions.create(
                                model=st.session_state["openai_model"],
                                messages=[
                                    {"role": "system", "content": "Analyze video frames."},
                                    {"role": "user", "content": [
                                        {"type": "image_url", 
                                         "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
                                        for frame in frames
                                    ]}
                                ]
                            )
                            st.markdown(response.choices[0].message.content)
            else:
                st.write("No videos found.")

    elif tab_main == "📝 Editor":
        if st.session_state.editing_file:
            st.subheader(f"Editing: {st.session_state.editing_file}")
            new_text = st.text_area("✏️ Content:", st.session_state.edit_new_content, height=300)
            if st.button("💾 Save"):
                with open(st.session_state.editing_file, 'w', encoding='utf-8') as f:
                    f.write(new_text)
                st.success("File updated successfully!")
                st.session_state.should_rerun = True
                st.session_state.editing_file = None
        else:
            st.write("Select a file from the sidebar to edit.")

    # Display file manager in sidebar
    display_file_manager_sidebar(groups_sorted)

    # Display viewed group content
    if st.session_state.viewing_prefix and any(st.session_state.viewing_prefix == group for group, _ in groups_sorted):
        st.write("---")
        st.write(f"**Viewing Group:** {st.session_state.viewing_prefix}")
        for group_name, files in groups_sorted:
            if group_name == st.session_state.viewing_prefix:
                for f in files:
                    fname = os.path.basename(f)
                    ext = os.path.splitext(fname)[1].lower().strip('.')
                    st.write(f"### {fname}")
                    if ext == "md":
                        content = open(f, 'r', encoding='utf-8').read()
                        st.markdown(content)
                    elif ext in ["mp3", "wav"]:
                        st.audio(f)
                    else:
                        st.markdown(get_download_link(f), unsafe_allow_html=True)
                break
        if st.button("❌ Close"):
            st.session_state.viewing_prefix = None
            st.session_state['marquee_content'] = "🚀 Welcome to TalkingAIResearcher | 🤖 Your Research Assistant"

    st.markdown("""
    <style>
        .main { background: linear-gradient(to right, #1a1a1a, #2d2d2d); color: #fff; }
        .stMarkdown { font-family: 'Helvetica Neue', sans-serif; }
        .stButton>button { margin-right: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.should_rerun:
        st.session_state.should_rerun = False
        st.rerun()

if __name__ == "__main__":
    main()
