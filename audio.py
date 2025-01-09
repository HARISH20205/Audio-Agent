import whisper

model = whisper.load_model("turbo")
result = model.transcribe("test.mp3")
print(result["text"])