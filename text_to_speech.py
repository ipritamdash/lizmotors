import edge_tts

async def text_to_speech(text, output_audio):
    communicate = edge_tts.Communicate(text)
    
    await communicate.save(output_audio)
