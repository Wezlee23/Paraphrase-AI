
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
    'ai_provider': 'openrouter',
    'openrouter_api_key': os.getenv('OPENROUTER_API_KEY', ''),
    'openrouter_endpoint': 'https://openrouter.ai/api/v1/chat/completions',
    'default_model': 'meta-llama/llama-3.3-70b-instruct:free',
    'local_endpoint': 'http://localhost:11434/v1/chat/completions',
    'local_model': 'llama3',
    'auto_save_interval': 1,
    'theme': 'dark',
    'prompt_humanize_sys': "You are a professional academic editor specializing in refining AI-generated text for academic audiences.",
    'prompt_humanize_user': "Rewrite this text to sound like it was written by an academic scholar. Use a formal, precise, and objective tone while ensuring high accuracy of the information. Avoid colloquialisms and maintain appropriate academic vocabulary. Keep all key information intact. Return only the rewritten text.",
    'prompt_summarize_sys': "You are an expert summarizer who condenses long texts into clear, concise bullet points.",
    'prompt_summarize_user': "Summarize this text into 3-7 key bullet points. Use simple language. Start each point with •",
    'prompt_grammar_sys': "You are a professional editor who improves writing clarity, grammar, and style.",
    'prompt_grammar_user': "Fix grammar, spelling, and improve clarity. Return only the improved text."
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
# AI HELPER FUNCTIONS
# ============================================

def call_openrouter_api(system_prompt, user_prompt, context=''):
    """Generic function to call OpenRouter API or local OpenAI-compatible endpoints with optional context"""
    config = get_config()
    provider = config.get('ai_provider', 'openrouter')
    
    if provider == 'local':
        api_key = "dummy"
        url = config.get('local_endpoint', 'http://localhost:11434/v1/chat/completions')
        model = config.get('local_model', 'llama3')
    else:
        api_key = config.get('openrouter_api_key') or os.getenv("OPENROUTER_API_KEY", "")
        url = config.get('openrouter_endpoint', 'https://openrouter.ai/api/v1/chat/completions')
        model = config.get('default_model', 'meta-llama/llama-3.3-70b-instruct:free')
    
    headers = {
        "Content-Type": "application/json"
    }
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        # Some local models (like LM Studio) might still expect a Bearer token format
        headers["Authorization"] = "Bearer dummy"
    
    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "user", "content": f"Context from uploaded files:\n{context[:4000]}"})
    messages.append({"role": "user", "content": user_prompt})
    
    data = {
        "model": model,
        "messages": messages
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=120)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            err_msg = f"Error: Unable to process request (Status: {response.status_code})"
            if response.status_code == 404 and provider == 'local':
                err_msg += f"\nHint: Ensure your local endpoint includes the full path (e.g., /v1/chat/completions if using Ollama's OpenAI compatibility layer). You entered: {url}"
            else:
                try:
                    err_msg += f" - {response.json().get('error', response.text)}"
                except:
                    err_msg += f" - {response.text}"
            return err_msg
    except requests.exceptions.Timeout:
        return "Error: The request timed out. Local models can take several minutes to generate a response depending on your hardware. Try a smaller model or check your server logs."
    except Exception as e:
        return f"Error: {str(e)}"


def humanize(text, context=''):
    config = get_config()
    system_prompt = config.get('prompt_humanize_sys', DEFAULT_CONFIG['prompt_humanize_sys'])
    user_prompt_template = config.get('prompt_humanize_user', DEFAULT_CONFIG['prompt_humanize_user'])
    if '{text}' in user_prompt_template:
        user_prompt = user_prompt_template.replace('{text}', text)
    else:
        user_prompt = f"{user_prompt_template}\n\nText:\n{text}"
    return call_openrouter_api(system_prompt, user_prompt, context)


def summarize_text(text, context=''):
    config = get_config()
    system_prompt = config.get('prompt_summarize_sys', DEFAULT_CONFIG['prompt_summarize_sys'])
    user_prompt_template = config.get('prompt_summarize_user', DEFAULT_CONFIG['prompt_summarize_user'])
    if '{text}' in user_prompt_template:
        user_prompt = user_prompt_template.replace('{text}', text)
    else:
        user_prompt = f"{user_prompt_template}\n\nText:\n{text}"
    return call_openrouter_api(system_prompt, user_prompt, context)



def improve_grammar(text, context=''):
    config = get_config()
    system_prompt = config.get('prompt_grammar_sys', DEFAULT_CONFIG['prompt_grammar_sys'])
    user_prompt_template = config.get('prompt_grammar_user', DEFAULT_CONFIG['prompt_grammar_user'])
    if '{text}' in user_prompt_template:
        user_prompt = user_prompt_template.replace('{text}', text)
    else:
        user_prompt = f"{user_prompt_template}\n\nText:\n{text}"
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



@app.route('/process', methods=['POST'])
def process_text():
    init_session()
    
    data = request.get_json()
    tool = data.get('tool', 'paraphrase')
    text = data.get('text', '')
    append_to_draft = data.get('append_to_draft', False)
    context = session.get('context', '')
    
    # Process based on tool
    if tool == 'paraphrase':
        result = humanize(text, context)
    elif tool == 'summarize':
        result = summarize_text(text, context)
    elif tool == 'grammar':
        result = improve_grammar(text, context)
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
            'ai_provider': data.get('ai_provider', 'openrouter'),
            'openrouter_api_key': data.get('openrouter_api_key', ''),
            'openrouter_endpoint': data.get('openrouter_endpoint', 'https://openrouter.ai/api/v1/chat/completions'),
            'default_model': data.get('default_model', 'mistralai/mistral-7b-instruct:free'),
            'local_endpoint': data.get('local_endpoint', 'http://localhost:11434/v1/chat/completions'),
            'local_model': data.get('local_model', 'llama3'),
            'auto_save_interval': max(1, min(60, int(data.get('auto_save_interval', 1)))),
            'theme': data.get('theme', 'dark'),
            'prompt_humanize_sys': data.get('prompt_humanize_sys', DEFAULT_CONFIG['prompt_humanize_sys']),
            'prompt_humanize_user': data.get('prompt_humanize_user', DEFAULT_CONFIG['prompt_humanize_user']),
            'prompt_summarize_sys': data.get('prompt_summarize_sys', DEFAULT_CONFIG['prompt_summarize_sys']),
            'prompt_summarize_user': data.get('prompt_summarize_user', DEFAULT_CONFIG['prompt_summarize_user']),
            'prompt_grammar_sys': data.get('prompt_grammar_sys', DEFAULT_CONFIG['prompt_grammar_sys']),
            'prompt_grammar_user': data.get('prompt_grammar_user', DEFAULT_CONFIG['prompt_grammar_user'])
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
