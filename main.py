import wave
import numpy as np
import pyaudio
import webrtcvad
import whisper
import time
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize VAD and Whisper model
vad = webrtcvad.Vad()
vad.set_mode(2)  # Sensitivity level: 0 (least sensitive) to 3 (most sensitive)
model = whisper.load_model("base")  # Load Whisper model

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)  # Replace with your Gemini API key

system_ins = """
You are a specialized assistant designed to break down complex instructions into simple, atomic actions. Given a sentence containing a detailed instruction, you must identify each simple, actionable step and return them as separate entries in a structured JSON format. Each step should be a single, independent action, such as "move," "turn," or "pick," and the output should be organized in a list.

Example Input: "To make the humanoid robot pick up an object, move forward, turn left, bend the arm, extend the hand, grip the object, lift the arm, move backward, and place the object down."

Expected Output:

json
Copy code
{
  "steps": [
    "Move forward",
    "Turn left",
    "Bend the arm",
    "Extend the hand",
    "Grip the object",
    "Lift the arm",
    "Move backward",
    "Place the object down"
  ]
}
"""

gemini_model = genai.GenerativeModel("gemini-1.5-flash",system_instruction=system_ins)

# Audio setup
RATE = 16000  # Sample rate
FRAME_DURATION = 20  # Frame duration in ms
CHUNK = int(RATE * FRAME_DURATION / 1000)  # 320 samples for 20 ms at 16 kHz
SILENCE_THRESHOLD = 10  # Number of silence frames before processing

def is_speech(audio_data, rate):
    """Check if audio data contains speech using WebRTC VAD."""
    audio_bytes = audio_data.tobytes()
    return vad.is_speech(audio_bytes, sample_rate=rate)

def record_and_transcribe():
    """Record audio, detect speech, and transcribe when speech ends."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    logging.info("Start talking...")
    audio_buffer = []
    silence_frames = 0

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            if is_speech(audio_data, RATE):
                audio_buffer.append(data)
                silence_frames = 0
                logging.info("Speech detected. Buffering audio...")
            else:
                silence_frames += 1
                logging.info("Silence detected. Incrementing silence frames...")

            if silence_frames >= SILENCE_THRESHOLD and audio_buffer:
                logging.info("Silence threshold reached. Processing audio...")
                save_and_transcribe(audio_buffer, RATE)
                audio_buffer = []  # Clear buffer
                logging.info("Audio buffer cleared. Waiting for next input...")
                time.sleep(10)  # Wait for 10 seconds before starting new input
                logging.info("Start talking...")
                silence_frames = 0
    except KeyboardInterrupt:
        logging.info("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

def save_and_transcribe(audio_buffer, rate):
    """Save audio buffer to a WAV file and transcribe it."""
    temp_wav = "temp_audio.wav"
    with wave.open(temp_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes (16-bit audio)
        wf.setframerate(rate)
        wf.writeframes(b''.join(audio_buffer))

    logging.info("Audio saved to temporary WAV file. Transcribing...")
    result = model.transcribe(temp_wav)
    transcription = result['text']
    logging.info(f"Transcription: {transcription}")

    if transcription:
        # Pass transcription to Gemini
        logging.info("Sending transcription to Gemini...")
        gemini_response = gemini_model.generate_content(transcription)
        logging.info(f"Gemini Response: {gemini_response.text}")

        # Print Gemini response
        print("Gemini Response:", gemini_response.text)

if __name__ == "__main__":
    record_and_transcribe()