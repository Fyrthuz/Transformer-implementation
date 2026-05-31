# dataset_preparation.py
import urllib.request
import os
import random

# =====================================================================
# 1. DESCARGA DEL DATASET CRUDO
# =====================================================================
data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data_path = "tinyshakespeare.txt"

if not os.path.exists(data_path):
    print("Descargando TinyShakespeare...")
    urllib.request.urlretrieve(data_url, data_path)

with open(data_path, 'r', encoding='utf-8') as f:
    raw_text = f.read()

# =====================================================================
# 2. TOKENIZER COMPACTO A NIVEL DE CARÁCTER
# =====================================================================
def extract_vocab(text_data):
    # Extrae los caracteres únicos reales presentes en la obra
    chars = sorted(list(set(text_data)))
    # Reservamos el ID 0 para <pad> por seguridad arquitectónica
    return ['<pad>'] + chars

# Extraemos el vocabulario global basado en todo el libro
vocab = extract_vocab(raw_text)

class SimpleTokenizer:
    def __init__(self, vocab_list):
        self.vocab_size = len(vocab_list)
        self.id_to_token = {i: char for i, char in enumerate(vocab_list)}
        self.token_to_id = {char: i for i, char in enumerate(vocab_list)}
        self.pad_token_id = self.token_to_id['<pad>']

    def encode(self, text):
        # Mapea carácter por carácter a su ID numérico
        return [self.token_to_id.get(char, self.pad_token_id) for char in text]

    def decode(self, token_ids):
        # Reconstruye el texto uniendo los caracteres individuales
        return "".join([self.id_to_token.get(i, '') for i in token_ids if i != self.pad_token_id])

# Instanciamos el tokenizador de caracteres
tokenizer = SimpleTokenizer(vocab)

# =====================================================================
# 3. SPLIT SEGURO Y SLIDING WINDOW POR CARACTERES
# =====================================================================
print("Tokenizando texto completo a nivel de carácter...")
all_tokens = tokenizer.encode(raw_text)

max_seq_len = 128

# Split físico e independiente del texto crudo tokenizado (90% / 10%)
split_idx = int(len(all_tokens) * 0.9)
train_tokens_raw = all_tokens[:split_idx]
test_tokens_raw = all_tokens[split_idx:]

# --- VENTANA DESLIZANTE PARA TRAIN (Paso de 16 caracteres) ---
# Al ser caracteres, avanzar de 16 en 16 genera una cantidad colosal de datos rústicos
train_stride = 16 
train_chunks = []
for i in range(0, len(train_tokens_raw) - max_seq_len, train_stride):
    train_chunks.append(train_tokens_raw[i : i + max_seq_len])

# --- BLOQUES LIMPIOS PARA TEST (Sin solapamiento) ---
test_stride = max_seq_len 
test_chunks = []
for i in range(0, len(test_tokens_raw) - max_seq_len, test_stride):
    test_chunks.append(test_tokens_raw[i : i + max_seq_len])

# Mezclamos train para eliminar la dependencia secuencial del libro
random.seed(42)
random.shuffle(train_chunks)

# Limitamos los conjuntos para que tu entrenamiento sea ágil (Experimento balanceado)
preprocessed_train_dataset = [{'text': chunk} for chunk in train_chunks[:35000]] 
preprocessed_test_dataset = [{'text': chunk} for chunk in test_chunks[:2000]]

print(f"--> ¡Estructura de Caracteres lista!")
print(f"    Tamaño del Vocabulario: {tokenizer.vocab_size} caracteres.")
print(f"    Bloques de Train (Stride 16): {len(preprocessed_train_dataset)}")
print(f"    Bloques de Test (Independiente): {len(preprocessed_test_dataset)}")