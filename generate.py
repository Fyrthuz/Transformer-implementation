# generate.py
import torch
from dataset_preparation import vocab, SimpleTokenizer
from transformer import Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Cargar Tokenizer y Configuración EXACTA del modelo entrenado
tokenizer = SimpleTokenizer(vocab)

config = {
    'vocab_size': tokenizer.vocab_size, 
    'd_model': 256,       
    'num_heads': 8,       
    'd_ff': 1024,         
    'num_layers': 4,      
    'dropout': 0.0 # Ponemos el dropout en 0 para generar texto
}

# 2. Inicializar arquitectura y cargar los pesos guardados
model = Transformer(**config).to(device)
model.load_state_dict(torch.load('best_model.pt', map_location=device))
model.eval()

# 3. Función de generación por muestreo con Temperatura
def generate_text(model, start_text, max_generated_chars=500, temperature=0.7):
    """
    start_text: El prompt inicial (ej. "ROMEO:\n")
    temperature: Controla la creatividad. 
                 - Cercano a 0.2: Muy conservador, casi no comete faltas pero se repite.
                 - Cercano a 0.7: Equilibrio perfecto entre sentido y variedad.
                 - Mayor a 1.0: Creativo pero empieza a inventar palabras locas.
    """
    context_ids = tokenizer.encode(start_text)
    input_tensor = torch.tensor([context_ids]).to(device) # Shape: (1, seq_len)
    
    generated_ids = list(context_ids)
    
    print(f"--- Generando texto a partir de: '{start_text}' ---")
    
    with torch.no_grad():
        for _ in range(max_generated_chars):
            # Si el contexto acumulado supera las 64 posiciones de tu modelo, recortamos el pasado
            if input_tensor.size(1) > 63:
                input_cond = input_tensor[:, -63:]
            else:
                input_cond = input_tensor
                
            # Crear máscara causal para el tamaño actual del input
            mask = torch.tril(torch.ones(input_cond.size(1), input_cond.size(1))).unsqueeze(0).to(device)
            
            # Obtener logits del último carácter de la secuencia
            logits = model(input_cond, mask=mask)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Aplicar Softmax para obtener probabilidades reales
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Muestrear en lugar de usar argmax para darle vida y sentido al texto
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # Acumular y actualizar tensores
            generated_ids.append(next_token_id)
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]]).to(device)], dim=1)
            
            # Imprimir carácter por carácter en tiempo real en la consola
            print(tokenizer.decode([next_token_id]), end='', flush=True)
            
    print("\n\n--- Fin del texto generado ---")

# =====================================================================
# EJECUCIÓN: PRUEBA TU MODELO AQUÍ
# =====================================================================
if __name__ == "__main__":
    # Dale un inicio dramático estilo Shakespeare
    prompt_inicial = "BAPTISTA:\n"
    generate_text(model, start_text=prompt_inicial, max_generated_chars=600, temperature=0.65)