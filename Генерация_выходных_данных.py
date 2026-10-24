# Промпт
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]

# Ввод данных
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=False
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Генерация контента
outputs = model.generate(**inputs, max_new_tokens=1024,)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Декодируем только новые токены и игнорируем специальные символы
processor.parse_response(response)
