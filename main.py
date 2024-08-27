import openai
import asyncio
import os
import time
import whisper
import webrtcvad
import soundfile as sf
import numpy as np
import edge_tts

def voice_to_text(audio_file):
    model = whisper.load_model("base")
    audio, sr = sf.read(audio_file)

    if sr != 16000:
        raise ValueError("Audio sample rate must be 16000 Hz")
    if len(audio.shape) > 1:
        raise ValueError("Audio must be mono")

    audio_int16 = (audio * 32767).astype(np.int16)

    vad = webrtcvad.Vad(1) 
    frame_duration = 20 
    frame_size = int(16000 * frame_duration / 1000)
    frames = [audio_int16[i:i + frame_size] for i in range(0, len(audio_int16), frame_size)]

    frames = [f for f in frames if len(f) == frame_size]

    active_frames = [f for f in frames if vad.is_speech(f.tobytes(), 16000)]

    active_audio = np.concatenate(active_frames)

    active_audio_float32 = active_audio.astype(np.float32) / 32767

    result = model.transcribe(active_audio_float32, language="en")
    text = result['text']

    return text

def text_to_llm(input_text):
    openai.api_key = "Your_OpenAPI_key"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Only respond with new information, and do not echo back the user's input. Also make the response in two lines only"},
            {"role": "user", "content": input_text}
        ],
        max_tokens=50,  
        n=1, 
        stop=["\n"], 
        temperature=0.7  
    )

    response_text = response['choices'][0]['message']['content']
    
    response_text = '. '.join(response_text.split('. ')[:2]) + '.'
    
    return response_text

async def text_to_speech(text, output_audio, voice="en-US-JennyNeural", rate="+0%"):
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicate.save(output_audio)

def main():
    audio_file = "inputmain.wav"
    transcribed_text = voice_to_text(audio_file)
    print(f"Transcribed Text: {transcribed_text}")
    
    llm_response = text_to_llm(transcribed_text)
    print(f"LLM Response: {llm_response}")
    
    output_dir = "../output"
    timestamp = int(time.time())
    output_audio = os.path.join(output_dir, f"output_audio_{timestamp}.mp3")
    
    asyncio.run(text_to_speech(llm_response, output_audio, voice="en-US-GuyNeural", rate="-5%"))
    print(f"Audio saved to: {output_audio}")

if __name__ == "__main__":
    main()
