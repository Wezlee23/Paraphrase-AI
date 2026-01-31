# Paraphrase AI

A powerful AI-powered learning workspace for humanizing, summarizing, organizing notes, improving grammar, and generating citations. Built with Flask and integrated with OpenRouter for access to free AI models.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Humanize Text** - Rewrite AI-generated content to sound natural and human-written
- **Summarize** - Condense long texts into clear, concise bullet points
- **Organize Notes** - Structure content into organized study notes with headers and formatting
- **Improve Grammar** - Fix grammar, spelling, and improve writing clarity
- **Generate Citations** - Extract and format sources in APA, MLA, or Chicago style

### Additional Features

- 📹 **YouTube Transcription** - Get transcripts from YouTube videos (with caption support or Whisper fallback)
- 📁 **File Upload** - Support for PDF, DOCX, TXT, and audio/video files
- 🎤 **Audio Transcription** - Transcribe uploaded audio/video using local Whisper or Groq API
- 📝 **Draft Builder** - Build and edit your draft as you work, with auto-save
- ⚙️ **Dynamic Model Selection** - Fetches available free models directly from OpenRouter
- 🎨 **Modern Dark UI** - Beautiful, responsive interface with resizable panels

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/flask_paraphraser.git
cd flask_paraphraser

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API key

# Run the application
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

## Requirements

- Python 3.8+
- OpenRouter API key (free tier available)
- Optional: Groq API key for audio transcription
- Optional: FFmpeg for YouTube audio download

## Documentation

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for detailed setup and configuration guide.

## License

MIT License - see LICENSE file for details.
