import wave
import numpy as np
import pyaudio
import webrtcvad
import whisper
import time

# Initialize VAD and Whisper model
vad = webrtcvad.Vad()
vad.set_mode(2)  # Sensitivity level: 0 (least sensitive) to 3 (most sensitive)
model = whisper.load_model("base")  # Load Whisper model

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

    print("Start talking...")
    audio_buffer = []
    silence_frames = 0

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            if is_speech(audio_data, RATE):
                audio_buffer.append(data)
                silence_frames = 0
            else:
                silence_frames += 1

            if silence_frames >= SILENCE_THRESHOLD and audio_buffer:
                print("Transcribing...")
                save_and_transcribe(audio_buffer, RATE)
                audio_buffer = []  # Clear buffer
                print("Please wait...")  # User-friendly message before starting new input
                time.sleep(2)  # Delay for 2 seconds
                print("Start talking...")
                silence_frames = 0
    except KeyboardInterrupt:
        print("Stopping...")
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

    # Transcribe audio using Whisper
    result = model.transcribe(temp_wav)
    print("Transcription:", result['text'])

if __name__ == "__main__":
    record_and_transcribe()
