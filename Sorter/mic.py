import whisper
import pyaudio
import wave
import numpy as np

model = whisper.load_model("tiny")  # Load model

def record_audio(seconds=5, filename="command.wav"):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    frames = []
    for _ in range(int(16000 / 1024 * seconds)):
        data = stream.read(1024)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(filename, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b"".join(frames))
    wf.close()

def recognize_command():
    record_audio()
    result = model.transcribe("command.wav", fp16=False)
    text = result["text"].lower()
    if "apple" in text:
        return "apple"
    elif "servo" in text:
        return "servo"
    return None

# Example usage
while True:
    obj = recognize_command()
    if obj:
        print(f"Command received: Give me {obj}")
        break