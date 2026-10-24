# Обновление pip необходимо для поддержки современных версий wheel-файлов
!pip install -q --upgrade pip

# Основные библиотеки: TensorFlow (бэкенд Keras), JAX/Flax (нативный формат Gemma),
# Transformers/HF Hub (для загрузки весов) и Pillow (для обработки изображений)
!pip install -q keras-nlp>=0.15.0 transformers sentencepiece pillow "tensorflow[and-cuda]" jax[cuda12_pip]>=0.6.0 flax

# Принудительный перезапуск ядра Python, чтобы изменения вступили в силу
import IPython
IPython.Application.instance().kernel.do_shutdown(True)
