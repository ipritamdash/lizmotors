# from transformers import pipeline

# def text_to_llm(input_text):
#     generator = pipeline("text-generation", model="gpt2")

#     response = generator(input_text, max_length=150, num_return_sequences=1)

#     return response[0]['generated_text']


from openai import OpenAI

def text_to_llm(input_text):
    OPENAI_API_KEY = "Your_OpenAPI_key"
    client = OpenAI(api_key = OPENAI_API_KEY)

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text}
        ]
    )

    return completion.choices[0].message["content"]

