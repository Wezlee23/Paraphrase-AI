# Setup Instructions

Complete guide to set up and run Paraphrase AI on your local machine.

## Prerequisites

- **Python 3.8+** - [Download Python](https://python.org/downloads/)
- **pip** - Python package manager (included with Python)
- **Git** - For cloning the repository

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/flask_paraphraser.git
cd flask_paraphraser
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here
GROQ_API_KEY=gsk_your-groq-key-here  # Optional, for audio transcription
SECRET_KEY=your-secret-key-here      # Optional, auto-generated if not set
```

#### Getting API Keys

1. **OpenRouter API Key** (Required)
   - Go to [openrouter.ai/keys](https://openrouter.ai/keys)
   - Create a free account
   - Generate an API key
   - Free tier includes access to multiple AI models

2. **Groq API Key** (Optional)
   - Go to [console.groq.com/keys](https://console.groq.com/keys)
   - Create a free account
   - Generate an API key
   - Used for fast audio transcription

## Running the Application

### Development Mode

```bash
python app.py
```

The application will start at `http://127.0.0.1:5000`

### Production Mode

For production, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Optional Features Setup

### YouTube Transcription with Whisper

For videos without captions, you can use local Whisper transcription:

```bash
# Install OpenAI Whisper
pip install openai-whisper

# Install yt-dlp for downloading audio
pip install yt-dlp

# Install FFmpeg (required for audio processing)
# Windows: Download from https://ffmpeg.org/download.html
# Linux: sudo apt install ffmpeg
# Mac: brew install ffmpeg
```

### PDF Support

```bash
pip install PyPDF2
```

### Word Document Support

```bash
pip install python-docx
```

## Configuration

Settings can be configured via the Settings page (`/settings`) or by editing `config.json`:

| Setting | Description | Default |
|---------|-------------|---------|
| `upload_folder` | Directory for uploaded files | `uploads` |
| `temp_folder` | Directory for temporary files | `temp_audio` |
| `max_file_size` | Maximum upload size (MB) | `100` |
| `default_model` | Default AI model | `meta-llama/llama-3.3-70b-instruct:free` |
| `whisper_model` | Local Whisper model size | `base` |
| `theme` | UI theme (dark/light) | `dark` |

## Project Structure

```
flask_paraphraser/
├── app.py              # Main Flask application
├── errors.py           # Error handling and logging
├── config.json         # Application configuration
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (create this)
├── static/
│   ├── css/            # Stylesheets
│   └── js/             # JavaScript files
├── templates/          # HTML templates
├── uploads/            # Uploaded files directory
├── temp_audio/         # Temporary audio files
└── logs/               # Application logs
```

## Troubleshooting

### Common Issues

1. **404 Error when processing text**
   - Check your OpenRouter API key is valid
   - Verify the selected model is available (refresh models in Settings)

2. **YouTube transcription fails**
   - Ensure `yt-dlp` is installed for videos without captions
   - Check if FFmpeg is installed and in PATH

3. **File upload fails**
   - Check `max_file_size` in settings
   - Ensure `uploads` directory exists and is writable

### Logs

Application logs are stored in the `logs/` directory with daily rotation.

## Support

For issues or feature requests, please open an issue on GitHub.
