from openai import OpenAI

# 注意端口是10088，不是10085
base_url = "http://0.0.0.0:10088/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)

# 测试问题
message = [
    {
        "role": "system",
        "content": "You are a helpful assistant that answers questions based on the knowledge graph.",
    },
    {
        "role": "user",
        "content": "Question: How is the U-value relevant to thermal insulation performance in glazing products?",
    }
]

response = client.chat.completions.create(
    model="llama",
    messages=message,
    max_tokens=2048,
    temperature=0.5
)
print("RAG回答:")
print(response.choices[0].message.content)