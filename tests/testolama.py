import ollama

response = ollama.chat(
    model='qwen2.5:7b',
    messages=[
        {'role': 'user', 'content': 'Trích xuất entity từ câu: "Ngô Sĩ Liên biên soạn Đại Việt Sử Ký Toàn Thư"'}
    ]
)
print(response['message']['content'])