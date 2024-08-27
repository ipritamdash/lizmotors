
import whisper
import webrtcvad
import soundfile as sf
import numpy as np

def voice_to_text(audio_file):
    model = whisper.load_model("base")
    audio, sr = sf.read(audio_file)

    print(f"Sample rate: {sr}, Channels: {len(audio.shape)}")

    if sr != 16000:
        raise ValueError("Audio sample rate must be 16000 Hz")
    if len(audio.shape) > 1:
        raise ValueError("Audio must be mono")

    audio_int16 = (audio * 32767).astype(np.int16)

    vad = webrtcvad.Vad(1)
    frame_duration = 20
    frame_size = int(16000 * frame_duration / 1000) 
    frames = [audio_int16[i:i + frame_size] for i in range(0, len(audio_int16) - frame_size + 1, frame_size)]

    print(f"Number of frames: {len(frames)}")

    frames = [f for f in frames if len(f) == frame_size]

    active_frames = []
    for f in frames:
        try:
            if vad.is_speech(f.tobytes(), 16000):
                active_frames.append(f)
        except Exception as e:
            print(f"Error while processing frame: {e}")

    if not active_frames:
        raise ValueError("No speech detected in audio file")

    active_audio = np.concatenate(active_frames)

    active_audio_float32 = active_audio.astype(np.float32) / 32767

    result = model.transcribe(active_audio_float32, language="en")
    text = result['text']

    return text
