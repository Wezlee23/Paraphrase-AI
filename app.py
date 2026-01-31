
from flask import Flask, render_template, request, session, redirect, url_for, send_file, jsonify
import subprocess
import uuid
import os
import re
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from errors import (
    setup_logging, register_error_handlers, setup_request_logging,
    AppError, APIError, FileProcessingError, TranscriptionError, ValidationError,
    error_response, success_response
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24).hex())

# ============================================
# CONFIGURATION MANAGEMENT
# ============================================

CONFIG_FILE = 'config.json'

DEFAULT_CONFIG = {
    'upload_folder': 'uploads',
    'temp_folder': 'temp_audio',
    'max_file_size': 100,
    'openrouter_api_key': os.getenv('OPENROUTER_API_KEY', ''),
    'openrouter_endpoint': 'https://openrouter.ai/api/v1/chat/completions',
    'groq_api_key': os.getenv('GROQ_API_KEY', ''),
    'default_model': 'meta-llama/llama-3.3-70b-instruct:free',
    'whisper_model': 'base',
    'auto_save_interval': 1,
    'theme': 'dark'
}


def load_config():
    """Load configuration from file or return defaults"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_config = json.load(f)
                # Merge with defaults to ensure all keys exist
                config = DEFAULT_CONFIG.copy()
                config.update(saved_config)
                return config
        except:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(config):
    """Save configuration to file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def get_config():
    """Get current configuration"""
    return load_config()


# Load initial config
config = load_config()

# Configure folders from config
UPLOAD_FOLDER = config['upload_folder']
TEMP_FOLDER = config['temp_folder']
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc', 'mp3', 'mp4', 'wav', 'webm', 'm4a', 'ogg'}

for folder in [UPLOAD_FOLDER, TEMP_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config['max_file_size'] * 1024 * 1024

# Initialize error handling and logging
setup_logging(app, log_dir='logs')
register_error_handlers(app)
setup_request_logging(app)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_type(filename):
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    if ext in {'txt'}:
        return 'text'
    elif ext in {'pdf'}:
        return 'pdf'
    elif ext in {'docx', 'doc'}:
        return 'document'
    elif ext in {'mp3', 'wav', 'm4a', 'ogg'}:
        return 'audio'
    elif ext in {'mp4', 'webm'}:
        return 'video'
    return 'unknown'


def init_session():
    """Initialize session with default values if not present"""
    if 'files' not in session:
        session['files'] = []
    if 'draft' not in session:
        session['draft'] = ''
    if 'context' not in session:
        session['context'] = ''
    if 'history' not in session:
        session['history'] = []


def extract_text_from_file(filepath, file_type):
    """Extract text content from various file types"""
    try:
        if file_type == 'text':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif file_type == 'pdf':
            try:
                import PyPDF2
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text() + '\n'
                    return text.strip()
            except ImportError:
                return "[PDF extraction requires PyPDF2. Install with: pip install PyPDF2]"
        elif file_type == 'document':
            try:
                from docx import Document
                doc = Document(filepath)
                return '\n'.join([para.text for para in doc.paragraphs])
            except ImportError:
                return "[DOCX extraction requires python-docx. Install with: pip install python-docx]"
        elif file_type in ['audio', 'video']:
            return "[Audio/Video file - click 'Transcribe' to extract text]"
    except Exception as e:
        return f"[Error extracting text: {str(e)}]"
    return ''


# ============================================
# TRANSCRIPTION FUNCTIONS (FREE METHODS)
# ============================================

def extract_youtube_video_id(url):
    """Extract video ID from various YouTube URL formats"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_youtube_transcript(video_url):
    """Get transcript from YouTube using youtube-transcript-api (FREE)"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
        
        video_id = extract_youtube_video_id(video_url)
        if not video_id:
            return None, "Could not extract video ID from URL. Make sure it's a valid YouTube link."
        
        try:
            # Simple approach: just get the transcript directly
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
            
            if transcript:
                # Combine all text segments
                full_text = ' '.join([entry['text'] for entry in transcript])
                return full_text, None
            
            return None, "No transcript content found"
            
        except TranscriptsDisabled:
            return None, "Transcripts are disabled for this video"
        except NoTranscriptFound:
            # Try to get any available transcript
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                for t in transcript_list:
                    transcript = t.fetch()
                    full_text = ' '.join([entry['text'] for entry in transcript])
                    return full_text, None
            except:
                pass
            return None, "No transcript available for this video. Try a video with captions enabled."
        except Exception as e:
            error_msg = str(e)
            if "Video unavailable" in error_msg:
                return None, "Video is unavailable or private"
            return None, f"Could not get transcript: {error_msg}"
            
    except ImportError:
        return None, "YouTube transcription requires youtube-transcript-api. Install with: pip install youtube-transcript-api"


def transcribe_audio_with_whisper(filepath):
    """Transcribe audio using local Whisper model (FREE, runs locally)"""
    try:
        import whisper
        
        # Load the base model (good balance of speed/accuracy)
        model = whisper.load_model("base")
        result = model.transcribe(filepath)
        return result["text"], None
        
    except ImportError:
        return None, "[Local transcription requires openai-whisper. Install with: pip install openai-whisper]"
    except Exception as e:
        return None, f"Transcription error: {str(e)}"


def transcribe_with_free_api(filepath):
    """Fallback: Use Groq's free Whisper API"""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return None, "No transcription API configured. Install openai-whisper locally or add GROQ_API_KEY."
    
    try:
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {groq_key}"}
        
        with open(filepath, 'rb') as audio_file:
            files = {'file': audio_file}
            data = {'model': 'whisper-large-v3'}
            response = requests.post(url, headers=headers, files=files, data=data)
        
        if response.status_code == 200:
            return response.json().get('text', ''), None
        else:
            return None, f"API error: {response.status_code}"
    except Exception as e:
        return None, f"API error: {str(e)}"


def transcribe_audio(filepath):
    """Main transcription function - tries multiple free methods"""
    # Try local Whisper first (completely free, no API needed)
    text, error = transcribe_audio_with_whisper(filepath)
    if text:
        return text, None
    
    # Fallback to Groq's free API if available
    text, error2 = transcribe_with_free_api(filepath)
    if text:
        return text, None
    
    return None, error or error2


# ============================================
# AI HELPER FUNCTIONS
# ============================================

def call_openrouter_api(system_prompt, user_prompt, context=''):
    """Generic function to call OpenRouter API with optional context"""
    config = get_config()
    api_key = config.get('openrouter_api_key') or os.getenv("OPENROUTER_API_KEY")
    url = config.get('openrouter_endpoint', 'https://openrouter.ai/api/v1/chat/completions')
    model = config.get('default_model', 'meta-llama/llama-3.3-70b-instruct:free')
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "user", "content": f"Context from uploaded files:\n{context[:4000]}"})
    messages.append({"role": "user", "content": user_prompt})
    
    data = {
        "model": model,
        "messages": messages
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            return f"Error: Unable to process request (Status: {response.status_code})"
    except Exception as e:
        return f"Error: {str(e)}"


def humanize(text, context=''):
    system_prompt = "You are a professional content editor specializing in rewriting AI-generated text to sound natural and human."
    user_prompt = f"""Rewrite this text to sound human-written. Use a natural, conversational tone with contractions. Keep all key information intact. Return only the rewritten text.

Text: "{text}"
"""
    return call_openrouter_api(system_prompt, user_prompt, context)


def summarize_text(text, context=''):
    system_prompt = "You are an expert summarizer who condenses long texts into clear, concise bullet points."
    user_prompt = f"""Summarize this text into 3-7 key bullet points. Use simple language. Start each point with •

Text: "{text}"
"""
    return call_openrouter_api(system_prompt, user_prompt, context)


def organize_notes(text, context=''):
    system_prompt = "You are a study skills expert who organizes notes into structured, easy-to-review formats."
    user_prompt = f"""Organize this text into structured study notes with:
- Clear section headers (use ## for headers)
- Bullet points for details
- Key terms in **bold**

Text: "{text}"
"""
    return call_openrouter_api(system_prompt, user_prompt, context)


def improve_grammar(text, context=''):
    system_prompt = "You are a professional editor who improves writing clarity, grammar, and style."
    user_prompt = f"""Fix grammar, spelling, and improve clarity. Return only the improved text.

Text: "{text}"
"""
    return call_openrouter_api(system_prompt, user_prompt, context)


def generate_citations(text, style="APA", context=''):
    system_prompt = "You are an academic citation expert who formats sources in various citation styles."
    user_prompt = f"""Extract sources from this text and format them in {style} style. If no sources found, say so.

Text: "{text}"
"""
    return call_openrouter_api(system_prompt, user_prompt, context)


# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    return redirect(url_for('workspace'))


@app.route('/workspace')
def workspace():
    init_session()
    return render_template('workspace.html', 
                         files=session.get('files', []),
                         draft=session.get('draft', ''),
                         context=session.get('context', ''))


@app.route('/upload', methods=['POST'])
def upload_files():
    init_session()
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    uploaded_files = request.files.getlist('files')
    new_files = []
    
    for file in uploaded_files:
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            file_type = get_file_type(filename)
            text_content = extract_text_from_file(filepath, file_type)
            
            file_info = {
                'id': uuid.uuid4().hex,
                'original_name': filename,
                'stored_name': unique_filename,
                'type': file_type,
                'uploaded_at': datetime.now().isoformat(),
                'text_content': text_content[:5000] if text_content else '',
                'size': os.path.getsize(filepath)
            }
            new_files.append(file_info)
            
            # Add to context if text was extracted
            if text_content and not text_content.startswith('['):
                session['context'] = session.get('context', '') + f"\n\n--- From {filename} ---\n{text_content[:2000]}"
    
    session['files'] = session.get('files', []) + new_files
    session.modified = True
    
    return jsonify({'success': True, 'files': new_files})


@app.route('/delete-file/<file_id>', methods=['POST'])
def delete_file(file_id):
    init_session()
    files = session.get('files', [])
    
    for i, f in enumerate(files):
        if f['id'] == file_id:
            # Delete physical file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f['stored_name'])
            if os.path.exists(filepath):
                os.remove(filepath)
            files.pop(i)
            break
    
    session['files'] = files
    session.modified = True
    return jsonify({'success': True})


@app.route('/transcribe-url', methods=['POST'])
def transcribe_url():
    """Transcribe video from YouTube URL (FREE using youtube-transcript-api)"""
    init_session()
    
    data = request.get_json()
    video_url = data.get('url', '')
    force_download = data.get('force_download', False)  # Force download even if captions exist
    
    if not video_url:
        return jsonify({'error': 'No URL provided'}), 400
    
    # Check if it's a YouTube URL
    if 'youtube.com' in video_url or 'youtu.be' in video_url:
        # First try to get existing captions (fastest method)
        if not force_download:
            transcript, error = get_youtube_transcript(video_url)
            if transcript:
                # Add to context
                session['context'] = session.get('context', '') + f"\n\n--- From YouTube Video ---\n{transcript[:3000]}"
                session.modified = True
                return jsonify({'success': True, 'transcript': transcript, 'method': 'captions'})
        
        # If captions not available or force_download is True, download and transcribe
        transcript, error = download_and_transcribe_youtube(video_url)
        if transcript:
            session['context'] = session.get('context', '') + f"\n\n--- From YouTube Video (Whisper) ---\n{transcript[:3000]}"
            session.modified = True
            return jsonify({'success': True, 'transcript': transcript, 'method': 'whisper'})
        else:
            return jsonify({'error': error}), 400
    else:
        return jsonify({'error': 'Currently only YouTube URLs are supported for transcription'}), 400


def download_and_transcribe_youtube(video_url):
    """Download YouTube audio and transcribe using Whisper (for videos without captions)"""
    audio_filepath = None
    try:
        # Generate unique filename for the audio
        audio_filename = f"yt_audio_{uuid.uuid4().hex}.mp3"
        audio_filepath = os.path.join(app.config['TEMP_FOLDER'], audio_filename)
        
        # Download audio using yt-dlp
        result = subprocess.run(
            [
                'yt-dlp', 
                '-x',  # Extract audio
                '--audio-format', 'mp3',
                '--audio-quality', '0',  # Best quality
                '-o', audio_filepath,
                '--no-playlist',  # Don't download playlists
                '--quiet',  # Less verbose
                video_url
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            return None, f"Failed to download audio: {result.stderr}"
        
        # yt-dlp might add extension, check for the file
        if not os.path.exists(audio_filepath):
            # Try with .mp3 extension if yt-dlp added it
            if os.path.exists(audio_filepath + '.mp3'):
                audio_filepath = audio_filepath + '.mp3'
            else:
                # Search for the file
                for f in os.listdir(app.config['TEMP_FOLDER']):
                    if f.startswith(f"yt_audio_{audio_filename.split('_')[2].split('.')[0]}"):
                        audio_filepath = os.path.join(app.config['TEMP_FOLDER'], f)
                        break
        
        if not os.path.exists(audio_filepath):
            return None, "Could not find downloaded audio file"
        
        # Transcribe the audio
        transcript, error = transcribe_audio(audio_filepath)
        
        if transcript:
            return transcript, None
        else:
            return None, error or "Transcription failed"
            
    except subprocess.TimeoutExpired:
        return None, "Download timed out. The video might be too long."
    except FileNotFoundError:
        return None, "yt-dlp not found. Install it with: pip install yt-dlp"
    except Exception as e:
        return None, f"Error: {str(e)}"
    finally:
        # Cleanup: remove temporary audio file
        if audio_filepath and os.path.exists(audio_filepath):
            try:
                os.remove(audio_filepath)
            except:
                pass


@app.route('/transcribe-file/<file_id>', methods=['POST'])
def transcribe_file(file_id):
    """Transcribe uploaded audio/video file"""
    init_session()
    files = session.get('files', [])
    
    for i, f in enumerate(files):
        if f['id'] == file_id:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f['stored_name'])
            
            if not os.path.exists(filepath):
                return jsonify({'error': 'File not found'}), 404
            
            transcript, error = transcribe_audio(filepath)
            
            if transcript:
                # Update file's text content
                files[i]['text_content'] = transcript
                session['files'] = files
                
                # Add to context
                session['context'] = session.get('context', '') + f"\n\n--- Transcribed from {f['original_name']} ---\n{transcript[:2000]}"
                session.modified = True
                
                return jsonify({'success': True, 'transcript': transcript})
            else:
                return jsonify({'error': error}), 400
    
    return jsonify({'error': 'File not found'}), 404


@app.route('/process', methods=['POST'])
def process_text():
    init_session()
    
    data = request.get_json()
    tool = data.get('tool', 'paraphrase')
    text = data.get('text', '')
    append_to_draft = data.get('append_to_draft', False)
    citation_style = data.get('citation_style', 'APA')
    
    context = session.get('context', '')
    
    # Process based on tool
    if tool == 'paraphrase':
        result = humanize(text, context)
    elif tool == 'summarize':
        result = summarize_text(text, context)
    elif tool == 'notes':
        result = organize_notes(text, context)
    elif tool == 'grammar':
        result = improve_grammar(text, context)
    elif tool == 'citations':
        result = generate_citations(text, citation_style, context)
    else:
        result = text
    
    # Update draft if requested
    if append_to_draft:
        current_draft = session.get('draft', '')
        session['draft'] = current_draft + '\n\n' + result if current_draft else result
        session.modified = True
    
    # Add to history
    history = session.get('history', [])
    history.append({
        'tool': tool,
        'input': text[:200],
        'output': result[:200],
        'timestamp': datetime.now().isoformat()
    })
    session['history'] = history[-20:]  # Keep last 20
    session.modified = True
    
    return jsonify({
        'success': True, 
        'result': result,
        'draft': session.get('draft', '')
    })


@app.route('/update-draft', methods=['POST'])
def update_draft():
    init_session()
    data = request.get_json()
    session['draft'] = data.get('draft', '')
    session.modified = True
    return jsonify({'success': True})


@app.route('/clear-workspace', methods=['POST'])
def clear_workspace():
    # Delete all uploaded files
    files = session.get('files', [])
    for f in files:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f['stored_name'])
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
    
    # Clear session
    session.clear()
    init_session()
    
    return jsonify({'success': True})


@app.route('/export-draft')
def export_draft():
    init_session()
    draft = session.get('draft', '')
    
    if not draft:
        return "No draft to export", 400
    
    # Create temp file
    filename = f"draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    filepath = os.path.join(app.config['TEMP_FOLDER'], filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(draft)
    
    return send_file(filepath, as_attachment=True, download_name=filename)


@app.route('/get-file-content/<file_id>')
def get_file_content(file_id):
    init_session()
    files = session.get('files', [])
    
    for f in files:
        if f['id'] == file_id:
            return jsonify({
                'success': True,
                'content': f.get('text_content', ''),
                'filename': f.get('original_name', '')
            })
    
    return jsonify({'error': 'File not found'}), 404


# ============================================
# SETTINGS ROUTES
# ============================================

@app.route('/api/models')
def get_free_models():
    """Fetch available free models from OpenRouter API"""
    try:
        response = requests.get('https://openrouter.ai/api/v1/models')
        if response.status_code == 200:
            data = response.json()
            models = data.get('data', [])
            
            # Filter for free models (those with :free suffix or zero pricing)
            free_models = []
            for model in models:
                model_id = model.get('id', '')
                pricing = model.get('pricing', {})
                prompt_price = float(pricing.get('prompt', '1') or '1')
                completion_price = float(pricing.get('completion', '1') or '1')
                
                # Check if model is free (has :free suffix or zero pricing)
                if ':free' in model_id or (prompt_price == 0 and completion_price == 0):
                    # Get a nice display name
                    name = model.get('name', model_id)
                    free_models.append({
                        'id': model_id,
                        'name': name,
                        'context_length': model.get('context_length', 0)
                    })
            
            # Sort by name
            free_models.sort(key=lambda x: x['name'].lower())
            
            return jsonify({'success': True, 'models': free_models})
        else:
            return jsonify({'success': False, 'error': f'API returned status {response.status_code}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/settings')
def settings():
    """Render settings page"""
    config = get_config()
    return render_template('settings.html', config=config)


@app.route('/settings/save', methods=['POST'])
def save_settings():
    """Save settings to config file"""
    try:
        data = request.get_json()
        
        # Validate and sanitize settings
        new_config = {
            'upload_folder': data.get('upload_folder', 'uploads'),
            'temp_folder': data.get('temp_folder', 'temp_audio'),
            'max_file_size': max(1, min(500, int(data.get('max_file_size', 100)))),
            'openrouter_api_key': data.get('openrouter_api_key', ''),
            'openrouter_endpoint': data.get('openrouter_endpoint', 'https://openrouter.ai/api/v1/chat/completions'),
            'groq_api_key': data.get('groq_api_key', ''),
            'default_model': data.get('default_model', 'mistralai/mistral-7b-instruct:free'),
            'whisper_model': data.get('whisper_model', 'base'),
            'auto_save_interval': max(1, min(60, int(data.get('auto_save_interval', 1)))),
            'theme': data.get('theme', 'dark')
        }
        
        # Create folders if they don't exist
        for folder in [new_config['upload_folder'], new_config['temp_folder']]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        # Update app config
        app.config['UPLOAD_FOLDER'] = new_config['upload_folder']
        app.config['TEMP_FOLDER'] = new_config['temp_folder']
        app.config['MAX_CONTENT_LENGTH'] = new_config['max_file_size'] * 1024 * 1024
        
        if save_config(new_config):
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Failed to save config file'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/settings/reset', methods=['POST'])
def reset_settings():
    """Reset settings to defaults"""
    try:
        # Remove config file to use defaults
        if os.path.exists(CONFIG_FILE):
            os.remove(CONFIG_FILE)
        
        # Reset app config
        app.config['UPLOAD_FOLDER'] = DEFAULT_CONFIG['upload_folder']
        app.config['TEMP_FOLDER'] = DEFAULT_CONFIG['temp_folder']
        app.config['MAX_CONTENT_LENGTH'] = DEFAULT_CONFIG['max_file_size'] * 1024 * 1024
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
