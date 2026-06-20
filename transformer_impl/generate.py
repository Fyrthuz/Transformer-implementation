import torch

def generate_text(model, tokenizer, prompt, max_chars=500, temperature=0.7, device='cpu'):
    model.eval()
    context_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([context_ids], device=device)

    print(f"--- Generating from: '{prompt}' ---")

    with torch.no_grad():
        for _ in range(max_chars):
            if input_tensor.size(1) > 128:
                input_cond = input_tensor[:, -128:]
            else:
                input_cond = input_tensor

            mask = model.generate_causal_mask(input_cond.size(1), device)
            logits = model(input_cond, mask=mask)
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

            input_tensor = torch.cat([input_tensor, torch.tensor([[next_id]], device=device)], dim=1)
            print(tokenizer.decode([next_id]), end='', flush=True)

    print("\n\n--- End of generated text ---")
