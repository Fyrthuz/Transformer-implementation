# train.py
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset_preparation import preprocessed_train_dataset, preprocessed_test_dataset, vocab, SimpleTokenizer
from transformer import Transformer

# 1. Hardware y TensorBoard
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter('runs/transformer_shakespeare_character_level')
print(f"Utilizando dispositivo: {device}")

# 2. Inicializar Tokenizer con el vocabulario real
tokenizer = SimpleTokenizer(vocab)

# 3. Configuración de la Arquitectura (Optimizada para caracteres)
config = {
    'vocab_size': tokenizer.vocab_size, # ¡Ahora es ~66 en lugar de 50,000!
    'd_model': 256,       
    'num_heads': 8,       
    'd_ff': 1024,          
    'num_layers': 4,      
    'dropout': 0.1        # Un nivel de dropout moderado y sano
}

def initialize_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.weight, 1.0)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0, std=config['d_model']**-0.5)

# 4. Hiperparámetros de Entrenamiento Estable
train_config = {
    'batch_size': 64,
    'num_epochs': 30,          # 30 épocas darán una convergencia profunda en caracteres
    'learning_rate': 0.0005,   # LR estándar muy estable para este tamaño de vocabulario
    'weight_decay': 0.01,     # Un poco de regularización para evitar overfitting en caracteres
    'loss_fn': torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=0.025   # Reducido levemente porque el vocabulario es pequeño
    ),
}

# 5. Dataset Adaptador
class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        self.data = []
        for example in dataset:
            # Dado que viene pre-tokenizado como enteros desde el script anterior, pasa directo
            tokens = example['text']
            if len(tokens) > max_len: tokens = tokens[:max_len]
            padded = tokens + [tokenizer.pad_token_id] * (max_len - len(tokens))
            self.data.append(torch.tensor(padded))
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

max_seq_len = 128
train_loader = DataLoader(TextDataset(preprocessed_train_dataset, tokenizer, max_seq_len), 
                          batch_size=train_config['batch_size'], shuffle=True)
test_loader = DataLoader(TextDataset(preprocessed_test_dataset, tokenizer, max_seq_len), 
                         batch_size=train_config['batch_size'])

# 6. Inicialización del Modelo
model = Transformer(**config).to(device)
model.apply(initialize_weights)

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=train_config['learning_rate'],
    weight_decay=train_config['weight_decay']
)
loss_fn = train_config['loss_fn']

scheduler = CosineAnnealingLR(optimizer, T_max=train_config['num_epochs'])

def generate_causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).to(device)

# 7. Bucle de Entrenamiento Causal
global_step = 0
best_test_loss = float('inf')
pad_id = tokenizer.pad_token_id

print("Iniciando entrenamiento a nivel de carácter...")
for epoch in range(train_config['num_epochs']):
    model.train()
    total_train_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        input_batch = batch.to(device)
        
        inputs = input_batch[:, :-1]
        targets = input_batch[:, 1:]
        
        mask = generate_causal_mask(inputs.size(1))
        
        output_batch = model(inputs, mask=mask)
        loss = loss_fn(output_batch.reshape(-1, config['vocab_size']), targets.reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_train_loss += loss.item()
        writer.add_scalar('Loss/train_step', loss.item(), global_step)
        global_step += 1
        
    scheduler.step()
    avg_train_loss = total_train_loss / len(train_loader)

    # --- Bucle de Evaluación ---
    model.eval()
    total_test_loss = 0
    
    with torch.no_grad():
        for i, t_batch in enumerate(test_loader):
            t_batch = t_batch.to(device)
            t_inputs = t_batch[:, :-1]
            t_targets = t_batch[:, 1:]
            
            t_mask = generate_causal_mask(t_inputs.size(1))
            t_output = model(t_inputs, mask=t_mask)
            
            t_loss = loss_fn(t_output.reshape(-1, config['vocab_size']), t_targets.reshape(-1))
            total_test_loss += t_loss.item()
            
            # Mostrar la evolución de la escritura en caracteres dentro de TensorBoard
            if i == 0:
                preds = t_output.argmax(dim=-1)
                clean_input = [tok for tok in t_inputs[0].tolist() if tok != pad_id]
                clean_target = [tok for tok in t_targets[0].tolist() if tok != pad_id]
                clean_preds = preds[0].tolist()[:len(clean_target)]
                
                test_example = f"**Input**: {tokenizer.decode(clean_input)}  \n" \
                               f"**Target**: {tokenizer.decode(clean_target)}  \n" \
                               f"**Predicted**: {tokenizer.decode(clean_preds)}"
                writer.add_text('Muestras/Prediccion_Test', test_example, epoch)

    avg_test_loss = total_test_loss / len(test_loader)
    
    # Registro y guardado del checkpoint real
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"  --> ¡Nuevo mejor modelo guardado por caracteres! (Test Loss: {avg_test_loss:.4f})")

    writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
    writer.add_scalar('Loss/test_epoch', avg_test_loss, epoch)
    writer.add_scalar('Params/Learning_Rate', scheduler.get_last_lr()[0], epoch)
    writer.flush()
    
    print(f"Epoch [{epoch + 1:02d}/{train_config['num_epochs']}] | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Test Loss: {avg_test_loss:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.6f}")

writer.close()
print("\n¡Entrenamiento completado!")


