import os
import subprocess
from flask import Flask, request, render_template, send_from_directory
import uuid # For generating unique filenames
import time # For delays in cleanup (optional)

app = Flask(__name__)

# Configure upload folder for temporary audio files
# It's good practice to create this directory if it doesn't exist
UPLOAD_FOLDER = 'temp_audio'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    transcript = None
    error = None

    if request.method == 'POST':
        video_url = request.form.get('video_url')
        if not video_url:
            error = "Please enter a video URL."
            return render_template('index.html', transcript=transcript, error=error)

        # Generate a unique filename for the audio to avoid conflicts
        audio_filename = f"audio_{uuid.uuid4().hex}.mp3"
        audio_filepath = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)

        try:
            # Step 1: Extract audio using yt-dlp
            # -x: extract audio
            # --audio-format mp3: convert audio to mp3 format
            # -o: output filename template
            # -q: quiet mode (less verbose output)
            print(f"Attempting to download audio from: {video_url}")
            subprocess.run(
                ['yt-dlp', '-x', '--audio-format', 'mp3', '-o', audio_filepath, video_url],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"Audio downloaded to: {audio_filepath}")

            # Step 2: Placeholder for AI Transcription
            # In a real application, you would send 'audio_filepath' to an AI transcription service.
            # Examples: Google Cloud Speech-to-Text, AWS Transcribe, OpenAI Whisper, AssemblyAI.
            # This part would involve:
            # 1. Authenticating with the AI service.
            # 2. Uploading the audio file or providing a link to it.
            # 3. Making an API call to start the transcription.
            # 4. Polling/waiting for the transcription to complete.
            # 5. Retrieving the transcribed text.

            # For demonstration, we'll return a mock transcript.
            transcript = (
                "Transcription simulation successful! \n\n"
                "This is where the actual transcribed text from the AI service would appear. "
                "For example: 'Hello, welcome to this video. Today we will be discussing web development "
                "with Flask and how to extract audio from video URLs using yt-dlp. "
                "The process involves several steps, from fetching the video to processing its audio. "
                "Thank you for watching!'"
            )
            print("Transcription simulated.")

        except subprocess.CalledProcessError as e:
            error = f"Error extracting audio: {e.stderr}"
            print(f"Subprocess Error: {e.stderr}")
        except Exception as e:
            error = f"An unexpected error occurred: {str(e)}"
            print(f"General Error: {str(e)}")
        finally:
            # Step 3: Clean up the temporary audio file
            if os.path.exists(audio_filepath):
                try:
                    os.remove(audio_filepath)
                    print(f"Cleaned up audio file: {audio_filepath}")
                except OSError as e:
                    print(f"Error deleting temporary file {audio_filepath}: {e}")

    return render_template('index.html', transcript=transcript, error=error)

if __name__ == '__main__':
    # To run this Flask app:
    # 1. Make sure you have Flask and yt-dlp installed:
    #    pip install Flask yt-dlp
    # 2. Save this code as 'app.py' in your project root.
    # 3. Create a 'templates' folder in the same directory as 'app.py'.
    # 4. Save the HTML code (provided next) as 'index.html' inside the 'templates' folder.
    # 5. Run this file: python app.py
    # 6. Open your web browser and go to http://127.0.0.1:5000/
    app.run(debug=True) # debug=True is good for development, disable in production
