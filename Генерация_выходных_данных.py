import textwrap
import torch

# Подключаем модуль для работы с выходным текстом
def wrap_content_builtin(response: str, width: int = 80) -> str:
    if not response:
        return ""
    wrapped_lines = textwrap.wrap(
        response,
        width=width,
        break_long_words=False,
        replace_whitespace=False
    )
    return "\n".join(wrapped_lines)

# --- Инициализация истории со СИСТЕМНЫМ ПРОМПТОМ ---
# Это сообщение добавляется один раз и задает модель поведения на всю сессию
chat_history = [{"role": "system", "content": "Ты - полезный помощник c долей сарказма"}]

print("Диалог начат. Введите 'exit' для завершения.")
while True:
    prompt = input("\nВы: ")
    if prompt.strip().lower() == 'exit':
        print('Чавакакава.')
        break

    chat_history.append({"role": "user", "content": prompt})

    # Применяем шаблон ко всей истории
    inputs = processor.apply_chat_template(
        chat_history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Фиксируем длину входного контекста ДО генерации
    # Так как inputs теперь является чистым Tensor, используем len()
    input_len = inputs.shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=1024,
            temperature=0.5,
            do_sample=True,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        )

    # Извлекаем только новые токены, используя сохраненное значение input_len
    new_tokens = outputs[0][input_len:]

    response_text = processor.decode(new_tokens, skip_special_tokens=True)

    chat_history.append({"role": "assistant", "content": response_text})

    print("\nГена: ")
    print(wrap_content_builtin(response_text, 80))
    print('-'*80)
