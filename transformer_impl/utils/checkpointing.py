import os
import torch


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None, global_step=None, metrics=None, config=None):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'global_step': global_step,
        'metrics': metrics or {},
        'config': config or {},
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    if optimizer and state.get('optimizer_state_dict'):
        optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler and state.get('scheduler_state_dict'):
        scheduler.load_state_dict(state['scheduler_state_dict'])
    return state


def load_checkpoint_with_adaptation(path, model, device='cpu'):
    """Load a checkpoint, adapting to architecture mismatches.

    Handles:
    - Different vocab_size: resizes embedding/output_layer, preserves overlap
    - Different num_layers: loads common prefix of layers
    - Different d_model / num_heads: warns and skips mismatched keys
    """
    state = torch.load(path, map_location=device, weights_only=True)
    ckpt = state.get('model_state_dict', state)

    model_sd = model.state_dict()
    loaded = 0
    skipped = 0
    resized = 0

    for key in ckpt:
        if key not in model_sd:
            skipped += 1
            continue

        ckpt_t = ckpt[key]
        model_t = model_sd[key]

        if ckpt_t.shape == model_t.shape:
            model_sd[key] = ckpt_t.to(device)
            loaded += 1
            continue

        if key in ('embedding.token_embedding.weight', 'output_layer.weight'):
            min_vocab = min(ckpt_t.size(0), model_t.size(0))
            min_dim = min(ckpt_t.size(1), model_t.size(1))
            model_sd[key][:min_vocab, :min_dim] = ckpt_t[:min_vocab, :min_dim]
            resized += 1
            continue

        if key == 'output_layer.bias':
            min_v = min(ckpt_t.size(0), model_t.size(0))
            model_sd[key][:min_v] = ckpt_t[:min_v]
            resized += 1
            continue

        skipped += 1

    model.load_state_dict(model_sd, strict=False)

    if loaded:
        print(f"  Checkpoint: loaded {loaded} layers matched")
    if resized:
        print(f"  Checkpoint: resized {resized} layers (vocab size differs)")
    if skipped:
        print(f"  Checkpoint: skipped {skipped} mismatched layers (architecture differs)")

    return state


def resume_training(path, model, optimizer, scheduler, device='cpu'):
    state = load_checkpoint(path, model, optimizer, scheduler, device)
    epoch = state.get('epoch', 0)
    global_step = state.get('global_step', 0)
    print(f"Resumed from {path}: epoch={epoch}, global_step={global_step}")
    return epoch, global_step, state.get('metrics', {})
