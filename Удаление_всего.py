# очистка всего
import IPython
from IPython.display import Javascript

# Принудительный запуск сборки мусора Python
import gc
gc.collect()

# Очистка кэша PyTorch/TensorFlow на GPU
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
except ImportError:
    pass

try:
    import tensorflow as tf
    tf.keras.backend.clear_session()
except ImportError:
    pass

# Главный инструмент Colab: жесткая очистка ядра со сбросом аллокатора CUDA
IPython.Application.instance().kernel.do_shutdown(True)

# Выводим сообщение для пользователя (появится ДО перезапуска ядра)
display(Javascript('''google.colab.output.setIframeHeight(0, true); alert("Очистка памяти запущена. Ядро будет перезапущено через пару секунд.");'''))
