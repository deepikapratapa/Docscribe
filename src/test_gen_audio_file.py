import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000  # Sample rate (Hz)
seconds = 5  # Duration

print("🎙️ Recording... Speak now!")
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()  # Wait until recording is finished
write("data/samples_audio/test.wav", fs, audio)
print("✅ Saved: data/samples_audio/test.wav")