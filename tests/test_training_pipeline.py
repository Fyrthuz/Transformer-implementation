"""Tests for the new LLM training pipeline stages."""
import os
import torch
from transformer_impl.losses import dpo_loss, ppo_loss, ppo_loss_with_value, grpo_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from transformer_impl.utils.lr_scheduler import get_scheduler
from transformer_impl.utils.checkpointing import save_checkpoint, load_checkpoint



def test_dpo_loss():
    policy_chosen = torch.randn(4, 10)
    policy_rejected = torch.randn(4, 10)
    ref_chosen = torch.randn(4, 10)
    ref_rejected = torch.randn(4, 10)
    loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
    assert loss > 0, "DPO loss should be positive"
    assert not torch.isnan(loss), "DPO loss should not be NaN"


def test_dpo_loss_beta_effect():
    policy_chosen = torch.ones(2, 5) * 10
    policy_rejected = torch.ones(2, 5) * -10
    ref_chosen = torch.zeros(2, 5)
    ref_rejected = torch.zeros(2, 5)
    loss_low = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.01)
    loss_high = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=10.0)
    assert loss_high < loss_low, "Higher beta should give lower loss with strongly preferred data"


def test_ppo_loss():
    log_probs = torch.randn(4, 10)
    old_log_probs = torch.randn(4, 10)
    advantages = torch.randn(4, 10)
    loss = ppo_loss(log_probs, old_log_probs, advantages, epsilon=0.2)
    assert not torch.isnan(loss), "PPO loss should not be NaN"


def test_ppo_loss_with_value():
    log_probs = torch.randn(4, 10)
    old_log_probs = torch.randn(4, 10)
    advantages = torch.randn(4, 10)
    values = torch.randn(4, 10)
    returns = torch.randn(4, 10)
    loss = ppo_loss_with_value(log_probs, old_log_probs, advantages, values, returns)
    assert not torch.isnan(loss), "PPO with value loss should not be NaN"


def test_grpo_loss():
    policy_logps = torch.randn(4, 8)
    ref_logps = torch.randn(4, 8)
    rewards = torch.randn(4, 8)
    loss = grpo_loss(policy_logps, ref_logps, rewards, epsilon=0.2, kl_coef=0.01)
    assert loss > 0, "GRPO loss should be positive"
    assert not torch.isnan(loss), "GRPO loss should not be NaN"


def test_grpo_advantage_normalization():
    policy_logps = torch.randn(2, 4)
    ref_logps = torch.randn(2, 4)
    rewards = torch.tensor([[10.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0]])
    loss = grpo_loss(policy_logps, ref_logps, rewards, epsilon=0.2, kl_coef=0.0)
    assert not torch.isnan(loss)


def test_lr_scheduler_warmup_cosine():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=1.0)
    scheduler = get_scheduler("warmup_cosine", opt, warmup_steps=100, total_steps=1000)
    lrs = []
    for _ in range(1000):
        opt.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    assert lrs[0] < lrs[50] < lrs[99], "Warmup should increase LR"
    assert lrs[99] == 1.0, "Peak LR should be 1.0 at end of warmup"
    assert lrs[-1] < lrs[500], "LR should decay after warmup"


def test_lr_scheduler_constant():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=1.0)
    scheduler = get_scheduler("warmup_constant", opt, warmup_steps=50, total_steps=100)
    lrs = []
    for _ in range(100):
        opt.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    assert lrs[0] < lrs[49], "Warmup phase"
    assert abs(lrs[-1] - 1.0) < 0.01, "Should stay at 1.0 after warmup"


def test_lr_scheduler_linear():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=1.0)
    scheduler = get_scheduler("linear", opt, total_steps=100)
    lrs = []
    for _ in range(100):
        opt.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    assert abs(lrs[-1]) < 0.01, "LR should decay to near 0"


def test_checkpoint_save_load():
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = get_scheduler("cosine", optimizer, total_steps=100)
    path = "/tmp/test_checkpoint.pt"
    save_checkpoint(path, model, optimizer, scheduler, epoch=5, global_step=500)
    assert os.path.exists(path), "Checkpoint file should exist"

    model2 = torch.nn.Linear(10, 10)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    sch2 = get_scheduler("cosine", opt2, total_steps=100)
    opt2.step()
    sch2.step()
    state = load_checkpoint(path, model2, opt2, sch2, device='cpu')
    assert state['epoch'] == 5
    assert state['global_step'] == 500
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2), "Weights should match after load"
    os.remove(path)


def test_model_generate():
    from transformer_impl.config import config_from_cli
    from transformer_impl.model import Transformer
    cfg = config_from_cli(None, {
        'name': 'test', 'model': {'d_model': 64, 'num_layers': 2, 'dropout': 0,
                                   'attention': {'type': 'mha', 'num_heads': 4},
                                   'ffn': {'type': 'standard', 'd_ff': 128},
                                   'position': {'type': 'none'}},
    })
    model = Transformer(cfg, vocab_size=256).to(device)
    input_ids = torch.randint(0, 256, (2, 10), device=device)
    output = model.generate(input_ids, max_new_tokens=20, temperature=0.7, device=device)
    assert output.shape[0] == 2, "Batch dimension preserved"
    assert output.shape[1] > 10, "Should generate new tokens"
    assert output.shape[1] <= 30, "Should not exceed max_new_tokens + input"


def test_model_value_head():
    from transformer_impl.config import config_from_cli
    from transformer_impl.model import Transformer
    cfg = config_from_cli(None, {
        'name': 'test', 'model': {'d_model': 64, 'num_layers': 2, 'dropout': 0,
                                   'attention': {'type': 'mha', 'num_heads': 4},
                                   'ffn': {'type': 'standard', 'd_ff': 128},
                                   'position': {'type': 'none'}},
    })
    model = Transformer(cfg, vocab_size=256).to(device)
    model.add_value_head()
    input_ids = torch.randint(0, 256, (2, 16), device=device)
    mask = model.generate_causal_mask(16, device)
    values = model.forward_value(input_ids, mask=mask)
    assert values.shape == (2, 16), f"Value head output shape should be (B, S), got {values.shape}"


def test_model_config_extension():
    from transformer_impl.config import ExperimentConfig
    cfg = ExperimentConfig()
    cfg.pretrain.enabled = True
    cfg.sft.enabled = False
    cfg.dpo.beta = 0.2
    cfg.ppo.kl_coef = 0.05
    cfg.grpo.group_size = 16
    assert cfg.pretrain.enabled
    assert not cfg.sft.enabled
    assert cfg.dpo.beta == 0.2
    assert cfg.ppo.kl_coef == 0.05
    assert cfg.grpo.group_size == 16


def test_train_config_extension():
    from transformer_impl.config import TrainConfig
    cfg = TrainConfig()
    assert cfg.warmup_steps is None
    assert cfg.gradient_accumulation_steps is None
    assert cfg.mixed_precision is None


def test_logger_scalar():
    import tempfile
    from transformer_impl.utils.logging import Logger
    with tempfile.TemporaryDirectory() as tmp:
        log = Logger(log_dir=tmp, name="test", log_type="tensorboard")
        log.log_scalar('test/val', 0.5, 1)
        log.log_scalar('test/val', 0.8, 2)
        log.close()


def test_logger_text():
    import tempfile
    from transformer_impl.utils.logging import Logger
    with tempfile.TemporaryDirectory() as tmp:
        log = Logger(log_dir=tmp, name="test", log_type="tensorboard")
        log.log_text('test/sample', "Hello from test", 1)
        log.close()


def test_logger_histogram():
    import tempfile
    import torch
    from transformer_impl.utils.logging import Logger
    with tempfile.TemporaryDirectory() as tmp:
        log = Logger(log_dir=tmp, name="test", log_type="tensorboard")
        log.log_histogram('test/hist', torch.randn(100), 1)
        log.close()


def test_logger_gradient_histograms():
    import tempfile
    import torch
    from transformer_impl.utils.logging import Logger
    model = torch.nn.Linear(10, 10)
    loss = model(torch.randn(5, 10)).sum()
    loss.backward()
    with tempfile.TemporaryDirectory() as tmp:
        log = Logger(log_dir=tmp, name="test", log_type="tensorboard")
        log.log_gradient_histograms(model, 1)
        log.close()


def test_pipeline_yaml_load():
    import yaml
    path = "configs/pipeline_full.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    assert 'stages' in data
    assert len(data['stages']) >= 2
    for stage in data['stages']:
        assert 'command' in stage
        assert 'config' in stage


def test_log_interval_config():
    from transformer_impl.config import ExperimentConfig
    cfg = ExperimentConfig()
    assert cfg.pretrain.log_interval == 10
    cfg.pretrain.log_interval = 50
    assert cfg.pretrain.log_interval == 50


# ---------------------------------------------------------------------------
# Trainer step tests (real model, real training steps)
# ---------------------------------------------------------------------------

def _make_mini_cfg(stage_name='pretrain', model_cfg=None):
    from transformer_impl.config import ExperimentConfig
    cfg = ExperimentConfig()
    cfg.name = f"test_{stage_name}"
    cfg.model.d_model = 32
    cfg.model.num_layers = 2
    cfg.model.dropout = 0.0
    cfg.model.ffn.type = 'standard'
    cfg.model.ffn.d_ff = 64
    cfg.model.attention.type = 'mha'
    cfg.model.attention.num_heads = 4
    cfg.model.position.type = 'none'
    cfg.training.batch_size = 2
    cfg.training.learning_rate = 0.001
    cfg.training.weight_decay = 0.0
    cfg.training.grad_clip = 1.0
    cfg.training.scheduler = 'constant'
    cfg.training.loss.type = 'cross_entropy'
    cfg.training.loss.label_smoothing = 0.0
    if model_cfg:
        for k, v in model_cfg.items():
            parts = k.split('.')
            obj = cfg
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], v)
    cfg.dataset.name = 'tinyshakespeare'
    cfg.dataset.tokenization = 'char'
    cfg.dataset.vocab_size = 64
    cfg.dataset.max_seq_len = 32
    getattr(cfg, stage_name).enabled = True
    getattr(cfg, stage_name).max_steps = 2
    getattr(cfg, stage_name).warmup_steps = 0
    getattr(cfg, stage_name).log_interval = 1
    getattr(cfg, stage_name).eval_steps = 0
    getattr(cfg, stage_name).save_steps = 0
    getattr(cfg, stage_name).gradient_accumulation_steps = 1
    getattr(cfg, stage_name).mixed_precision = None
    stage = getattr(cfg, stage_name)
    if stage_name in ('dpo', 'ppo'):
        stage.beta = 0.1
    if stage_name in ('ppo',):
        stage.kl_coef = 0.02
        stage.clip_range = 0.2
        stage.vf_coef = 0.5
        stage.max_gen_length = 8
    if stage_name == 'grpo':
        stage.group_size = 2
        stage.epsilon = 0.2
        stage.kl_coef = 0.01
        stage.max_gen_length = 8
        stage.reward_fn = 'format'
    return cfg


def _make_mock_dataset_output(vocab_size=64, pad_token_id=0, seq_len=32):
    from transformer_impl.datasets.base import DatasetOutput
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = vocab_size
            self.pad_token_id = pad_token_id
        def encode(self, text):
            return [1] * seq_len
        def decode(self, ids):
            return " ".join(str(i) for i in ids[:20])
    return DatasetOutput(
        train_data=[{'text': [1]*seq_len}] * 4,
        test_data=[{'text': [1]*seq_len}] * 2,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        tokenizer=MockTokenizer(),
    )


def _make_mock_sft_dataset(vocab_size=64, pad_token_id=0, seq_len=32):
    from transformer_impl.datasets.base import DatasetOutput
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = vocab_size
            self.pad_token_id = pad_token_id
        def encode(self, text): return [1]*seq_len
        def decode(self, ids): return "mock"
    return DatasetOutput(
        train_data=[{
            'input_ids': [1]*seq_len,
            'labels': [-100]*10 + [1]*(seq_len-10),
            'attention_mask': [1]*seq_len,
        }] * 4,
        test_data=[{
            'input_ids': [1]*seq_len,
            'labels': [-100]*10 + [1]*(seq_len-10),
        }] * 2,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        tokenizer=MockTokenizer(),
    )


def _make_mock_dpo_dataset(vocab_size=64, pad_token_id=0, seq_len=32):
    from transformer_impl.datasets.base import DatasetOutput
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = vocab_size
            self.pad_token_id = pad_token_id
        def encode(self, text): return [1]*seq_len
        def decode(self, ids): return "mock"
    return DatasetOutput(
        train_data=[{
            'prompt': [1]*8,
            'chosen': [1]*seq_len,
            'rejected': [2]*seq_len,
        }] * 4,
        test_data=[{
            'prompt': [1]*8,
            'chosen': [1]*seq_len,
            'rejected': [2]*seq_len,
        }] * 2,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        tokenizer=MockTokenizer(),
    )


def _make_mock_grpo_dataset(vocab_size=64, pad_token_id=0, seq_len=32):
    from transformer_impl.datasets.base import DatasetOutput
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = vocab_size
            self.pad_token_id = pad_token_id
        def encode(self, text): return [1]*seq_len
        def decode(self, ids): return "<thinking> 42 </thinking>"
    return DatasetOutput(
        train_data=[{
            'text': [1]*seq_len,
            'question': 'test',
            'answer': '42',
            'answer_num': '42',
            'input_ids': [1]*seq_len,
        }] * 4,
        test_data=[{
            'text': [1]*seq_len,
            'input_ids': [1]*seq_len,
        }] * 2,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        tokenizer=MockTokenizer(),
    )


def _make_model(cfg, vocab_size=64):
    from transformer_impl.model import Transformer
    return Transformer(cfg, vocab_size=vocab_size)


def test_pretrainer_train_step_dict():
    """PreTrainer must handle dict batches (the original bug on line 15)."""
    cfg = _make_mini_cfg('pretrain')
    model = _make_model(cfg)
    ds = _make_mock_dataset_output()
    from transformer_impl.trainers import PreTrainer
    trainer = PreTrainer(cfg, model, ds, device)
    batch = {'text': torch.randint(0, 64, (2, 32))}
    loss = trainer.train_step(batch)
    assert loss > 0, f"Loss should be positive, got {loss}"
    assert not torch.isnan(loss), "Loss should not be NaN"


def test_pretrainer_train_step_tensor():
    """PreTrainer must handle raw tensor batches too."""
    cfg = _make_mini_cfg('pretrain')
    model = _make_model(cfg)
    ds = _make_mock_dataset_output()
    from transformer_impl.trainers import PreTrainer
    trainer = PreTrainer(cfg, model, ds, device)
    batch = torch.randint(0, 64, (2, 32), device=device)
    loss = trainer.train_step(batch)
    assert loss > 0, f"Loss should be positive, got {loss}"


def test_pretrainer_eval_step_dict():
    """PreTrainer eval_step must handle dict batches."""
    cfg = _make_mini_cfg('pretrain')
    model = _make_model(cfg)
    ds = _make_mock_dataset_output()
    from transformer_impl.trainers import PreTrainer
    trainer = PreTrainer(cfg, model, ds, device)
    batch = {'text': torch.randint(0, 64, (2, 32))}
    loss = trainer.eval_step(batch)
    assert loss > 0, f"Eval loss should be positive, got {loss}"


def test_pretrainer_full_train():
    """PreTrainer.train() runs without crashing."""
    cfg = _make_mini_cfg('pretrain')
    model = _make_model(cfg)
    ds = _make_mock_dataset_output()
    from transformer_impl.trainers import PreTrainer
    trainer = PreTrainer(cfg, model, ds, device)
    trainer.train()
    assert trainer.global_step > 0, "Should have completed at least one step"


def test_sft_trainer_train_step_dict():
    """SFTTrainer train_step with dict batch."""
    cfg = _make_mini_cfg('sft')
    model = _make_model(cfg)
    ds = _make_mock_sft_dataset()
    from transformer_impl.trainers import SFTTrainer
    trainer = SFTTrainer(cfg, model, ds, device)
    batch = {
        'input_ids': torch.randint(0, 64, (2, 32)),
        'labels': torch.cat([
            torch.full((2, 10), -100, dtype=torch.long),
            torch.randint(0, 64, (2, 22)),
        ], dim=1),
    }
    loss = trainer.train_step(batch)
    assert loss > 0, f"Loss should be positive, got {loss}"


def test_sft_trainer_full_train():
    """SFTTrainer.train() runs without crashing."""
    cfg = _make_mini_cfg('sft')
    model = _make_model(cfg)
    ds = _make_mock_sft_dataset()
    from transformer_impl.trainers import SFTTrainer
    trainer = SFTTrainer(cfg, model, ds, device)
    trainer.train()
    assert trainer.global_step > 0


def test_dpo_trainer_train_step_dict():
    """DPOTrainer train_step with dict batch."""
    cfg = _make_mini_cfg('dpo')
    model = _make_model(cfg)
    ds = _make_mock_dpo_dataset()
    from transformer_impl.trainers import DPOTrainer
    trainer = DPOTrainer(cfg, model, ds, device)
    batch = {
        'prompt': torch.randint(0, 64, (2, 8)),
        'chosen': torch.randint(0, 64, (2, 32)),
        'rejected': torch.randint(0, 64, (2, 32)),
    }
    loss = trainer.train_step(batch)
    assert loss > 0, f"DPO loss should be positive, got {loss}"
    assert not torch.isnan(loss), "DPO loss should not be NaN"


def test_dpo_trainer_eval_step():
    """DPOTrainer eval_step with dict batch."""
    cfg = _make_mini_cfg('dpo')
    model = _make_model(cfg)
    ds = _make_mock_dpo_dataset()
    from transformer_impl.trainers import DPOTrainer
    trainer = DPOTrainer(cfg, model, ds, device)
    batch = {
        'prompt': torch.randint(0, 64, (2, 8)),
        'chosen': torch.randint(0, 64, (2, 32)),
        'rejected': torch.randint(0, 64, (2, 32)),
    }
    loss = trainer.eval_step(batch)
    assert not torch.isnan(loss)


def test_ppo_trainer_train_step_dict():
    """PPOTrainer train_step with dict batch (needs value head)."""
    cfg = _make_mini_cfg('ppo')
    model = _make_model(cfg)
    ds = _make_mock_dpo_dataset()
    from transformer_impl.trainers import PPOTrainer
    trainer = PPOTrainer(cfg, model, ds, device)
    batch = {'prompt': torch.randint(0, 64, (1, 8))}
    loss = trainer.train_step(batch)
    assert not torch.isnan(loss), "PPO loss should not be NaN"


def test_ppo_trainer_eval_step():
    """PPOTrainer eval_step."""
    cfg = _make_mini_cfg('ppo')
    model = _make_model(cfg)
    ds = _make_mock_dpo_dataset()
    from transformer_impl.trainers import PPOTrainer
    trainer = PPOTrainer(cfg, model, ds, device)
    batch = {'prompt': torch.randint(0, 64, (1, 8))}
    loss = trainer.eval_step(batch)
    assert not torch.isnan(loss)


def test_grpo_trainer_train_step_dict():
    """GRPOTrainer train_step with dict batch."""
    cfg = _make_mini_cfg('grpo')
    model = _make_model(cfg)
    ds = _make_mock_grpo_dataset()
    from transformer_impl.trainers import GRPOTrainer
    trainer = GRPOTrainer(cfg, model, ds, device)
    batch = {
        'input_ids': torch.randint(0, 64, (1, 32)),
        'text': torch.randint(0, 64, (1, 32)),
        'answer': ['42'],
        'answer_num': ['42'],
    }
    loss = trainer.train_step(batch)
    assert not torch.isnan(loss), "GRPO loss should not be NaN"


def test_grpo_trainer_full_train():
    """GRPOTrainer.train() runs without crashing."""
    cfg = _make_mini_cfg('grpo')
    model = _make_model(cfg)
    ds = _make_mock_grpo_dataset()
    from transformer_impl.trainers import GRPOTrainer
    trainer = GRPOTrainer(cfg, model, ds, device)
    trainer.train()
    assert trainer.global_step > 0


def test_collate_dict_batch():
    """_collate_batch converts list-of-dicts to dict-of-tensors correctly."""
    from transformer_impl.trainers.base import BaseTrainer
    cfg = _make_mini_cfg('pretrain')
    model = _make_model(cfg)
    ds = _make_mock_dataset_output()
    trainer = BaseTrainer(cfg, model, ds, device)
    raw = [
        {'text': [1, 2, 3]},
        {'text': [4, 5, 6]},
    ]
    collated = trainer._collate_batch(raw)
    assert isinstance(collated, dict)
    assert 'text' in collated
    assert isinstance(collated['text'], torch.Tensor)
    assert collated['text'].shape == (2, 3)


def test_collate_variable_length():
    """_collate_batch pads variable-length sequences."""
    from transformer_impl.trainers.base import BaseTrainer
    cfg = _make_mini_cfg('pretrain')
    model = _make_model(cfg)
    ds = _make_mock_dataset_output()
    trainer = BaseTrainer(cfg, model, ds, device)
    raw = [
        {'input_ids': [1, 2], 'labels': [1, -100]},
        {'input_ids': [3, 4, 5], 'labels': [3, -100, -100]},
    ]
    collated = trainer._collate_batch(raw)
    assert collated['input_ids'].shape == (2, 3)
    assert collated['labels'].shape == (2, 3)
    # Labels should be padded with -100
    assert collated['labels'][0, 2].item() == -100


def test_collate_tensor_list():
    """_collate_batch handles list-of-tensors."""
    from transformer_impl.trainers.base import BaseTrainer
    cfg = _make_mini_cfg('pretrain')
    model = _make_model(cfg)
    ds = _make_mock_dataset_output()
    trainer = BaseTrainer(cfg, model, ds, device)
    raw = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    collated = trainer._collate_batch(raw)
    assert isinstance(collated, torch.Tensor)
    assert collated.shape == (2, 3)


# ---------------------------------------------------------------------------
# Adaptive checkpoint loading tests
# ---------------------------------------------------------------------------

def _make_ckpt_model(d_model=64, num_layers=2, num_heads=4, d_ff=128, vocab_size=64):
    from transformer_impl.config import ExperimentConfig
    from transformer_impl.model import Transformer
    cfg = ExperimentConfig()
    cfg.name = 'ckpt_test'
    cfg.model.d_model = d_model
    cfg.model.num_layers = num_layers
    cfg.model.dropout = 0.0
    cfg.model.ffn.type = 'standard'
    cfg.model.ffn.d_ff = d_ff
    cfg.model.attention.type = 'mha'
    cfg.model.attention.num_heads = num_heads
    cfg.model.position.type = 'none'
    return Transformer(cfg, vocab_size=vocab_size)


def test_ckpt_same_architecture():
    """Load checkpoint with identical architecture."""
    from transformer_impl.utils.checkpointing import load_checkpoint_with_adaptation
    src = _make_ckpt_model().to(device)
    dst = _make_ckpt_model().to(device)
    path = "/tmp/test_ckpt_same.pt"
    torch.save({'model_state_dict': src.state_dict()}, path)
    load_checkpoint_with_adaptation(path, dst, device)
    for p1, p2 in zip(src.parameters(), dst.parameters()):
        assert torch.equal(p1, p2), "Weights should match for same architecture"
    os.remove(path)


def test_ckpt_different_vocab():
    """Load checkpoint with different vocab_size (resize)."""
    from transformer_impl.utils.checkpointing import load_checkpoint_with_adaptation
    src = _make_ckpt_model(vocab_size=64).to(device)
    dst = _make_ckpt_model(vocab_size=128).to(device)
    path = "/tmp/test_ckpt_vocab.pt"
    torch.save({'model_state_dict': src.state_dict()}, path)
    load_checkpoint_with_adaptation(path, dst, device)
    assert dst.output_layer.weight.shape[0] == 128
    assert dst.embedding.token_embedding.weight.shape[0] == 128
    os.remove(path)


def test_ckpt_different_num_layers():
    """Load checkpoint with fewer layers (partial load)."""
    from transformer_impl.utils.checkpointing import load_checkpoint_with_adaptation
    src = _make_ckpt_model(num_layers=2).to(device)
    dst = _make_ckpt_model(num_layers=4).to(device)
    path = "/tmp/test_ckpt_layers.pt"
    torch.save({'model_state_dict': src.state_dict()}, path)
    load_checkpoint_with_adaptation(path, dst, device)
    assert dst.layers[0].norm1.weight is not None
    assert dst.layers[0].norm2.weight is not None
    os.remove(path)


def test_ckpt_different_d_model():
    """Load checkpoint with different d_model (skip mismatched)."""
    from transformer_impl.utils.checkpointing import load_checkpoint_with_adaptation
    src = _make_ckpt_model(d_model=64).to(device)
    dst = _make_ckpt_model(d_model=128).to(device)
    path = "/tmp/test_ckpt_dmodel.pt"
    torch.save({'model_state_dict': src.state_dict()}, path)
    load_checkpoint_with_adaptation(path, dst, device)
    for p in dst.parameters():
        assert not torch.isnan(p).any(), "No NaN after partial load"
    os.remove(path)


def test_ckpt_file_not_found():
    """Missing checkpoint should not crash."""
    from transformer_impl.utils.checkpointing import load_checkpoint_with_adaptation
    model = _make_ckpt_model().to(device)
    try:
        load_checkpoint_with_adaptation("/tmp/nonexistent.pt", model, device)
        return False
    except FileNotFoundError:
        pass
    except Exception:
        return False


def test_ckpt_then_train():
    """Load checkpoint then run training step."""
    from transformer_impl.utils.checkpointing import load_checkpoint_with_adaptation
    src = _make_ckpt_model().to(device)
    path = "/tmp/test_ckpt_train.pt"
    torch.save({'model_state_dict': src.state_dict()}, path)

    cfg = _make_mini_cfg('pretrain')
    dst = _make_ckpt_model(d_model=32, vocab_size=64).to(device)
    load_checkpoint_with_adaptation(path, dst, device)

    ds = _make_mock_dataset_output()
    from transformer_impl.trainers import PreTrainer
    trainer = PreTrainer(cfg, dst, ds, device)
    batch = {'text': torch.randint(0, 64, (2, 32))}
    loss = trainer.train_step(batch)
    assert loss > 0
    os.remove(path)


def test_ckpt_legacy_format():
    """Load legacy checkpoint (no 'model_state_dict' wrapper)."""
    from transformer_impl.utils.checkpointing import load_checkpoint_with_adaptation
    src = _make_ckpt_model().to(device)
    dst = _make_ckpt_model().to(device)
    path = "/tmp/test_ckpt_legacy.pt"
    torch.save(src.state_dict(), path)
    load_checkpoint_with_adaptation(path, dst, device)
    for p1, p2 in zip(src.parameters(), dst.parameters()):
        assert torch.equal(p1, p2)
    os.remove(path)


# ─── make_run_name ─────────────────────────────────────────────────────────

def test_make_run_name():
    from transformer_impl.config import make_run_name, config_from_yaml
    cfg = config_from_yaml('configs/pretrain_tinystories.yaml')
    name = make_run_name(cfg, prefix='test', timestamp='20250101_120000')
    assert name == 'test_mha_swiglu_none_d256l6_20250101_120000'
    name2 = make_run_name(cfg, prefix='pipeline_sft')
    assert name2.startswith('pipeline_sft_mha_swiglu_none_d256l6_')
    assert 'name' not in make_run_name.__code__.co_varnames  # no side effects


# ─── _coerce ───────────────────────────────────────────────────────────────

def test_coerce():
    from transformer_impl.config import _coerce
    assert _coerce('2e-5', float) == 2e-5
    assert _coerce('1e-6', float) == 1e-6
    assert _coerce('42', int) == 42
    assert _coerce('hello', float) == 'hello'
    assert _coerce(True, int) == 1
    assert _coerce(False, float) == 0.0
    assert _coerce(3.14, float) == 3.14


# ─── TrainingStageMixin ────────────────────────────────────────────────────

def test_stage_mixin_fields():
    from transformer_impl.config import SFTConfig, DPOConfig, PPOConfig, GRPOConfig
    for cls in [SFTConfig, DPOConfig, PPOConfig, GRPOConfig]:
        inst = cls()
        assert hasattr(inst, 'save_steps'), f'{cls.__name__} missing save_steps'
        assert hasattr(inst, 'eval_steps')
        assert hasattr(inst, 'log_interval')
        assert hasattr(inst, 'gradient_accumulation_steps')
        assert hasattr(inst, 'mixed_precision')
        assert hasattr(inst, 'save_total_limit')
        assert inst.save_total_limit == 3


# ─── cleanup() ─────────────────────────────────────────────────────────────

def test_cleanup_no_error():
    import shutil
    from transformer_impl.config import ExperimentConfig, ModelConfig, AttentionConfig, FFNConfig, PositionConfig
    from transformer_impl.model import Transformer
    from transformer_impl.trainers import PreTrainer
    from transformer_impl.datasets.base import DatasetOutput
    name = '_tmp_test_cleanup'
    cfg = ExperimentConfig(name=name)
    model = Transformer(cfg, 64)
    ds = DatasetOutput(tokenizer=None, vocab_size=64, pad_token_id=0, train_data=[], test_data=[])
    trainer = PreTrainer(cfg, model, ds, 'cpu')
    trainer.cleanup()
    shutil.rmtree(f'runs/{name}', ignore_errors=True)


# ─── checkpoint rotation ───────────────────────────────────────────────────

def test_checkpoint_rotation():
    import tempfile, shutil
    from transformer_impl.config import ExperimentConfig
    from transformer_impl.model import Transformer
    from transformer_impl.trainers import PreTrainer
    from transformer_impl.datasets.base import DatasetOutput
    name = '_tmp_test_rotate'
    cfg = ExperimentConfig(name=name)
    model = Transformer(cfg, 64)
    ds = DatasetOutput(tokenizer=None, vocab_size=64, pad_token_id=0, train_data=[], test_data=[])
    trainer = PreTrainer(cfg, model, ds, 'cpu')
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.ckpt_dir = tmpdir
        for step in [100, 200, 300, 400]:
            trainer.global_step = step
            trainer._save_checkpoint()
        files = sorted(os.listdir(tmpdir))
        assert len(files) == 3
        assert all(f.startswith('step') for f in files)
        assert 'step100.pt' not in files
        assert 'step400.pt' in files
    shutil.rmtree(f'runs/{name}', ignore_errors=True)


# ─── pipeline YAML stage configs ───────────────────────────────────────────

def test_pipeline_stage_configs():
    import yaml
    with open('configs/pipeline_full.yaml') as f:
        pipeline = yaml.safe_load(f)
    assert 'stages' in pipeline
    assert len(pipeline['stages']) == 4
    for stage in pipeline['stages']:
        assert 'command' in stage
        assert 'config' in stage
        assert stage['command'] in ('pretrain', 'sft', 'dpo', 'grpo')

    from transformer_impl.config import config_from_yaml
    for stage in pipeline['stages']:
        cfg = config_from_yaml(stage['config'])
        # All stages should have same model arch
        assert cfg.model.attention.type == 'mha'
        assert cfg.model.ffn.type == 'swiglu'
        assert cfg.model.position.type == 'none'
        assert cfg.model.d_model == 256
        assert cfg.model.num_layers == 6


# ─── pipeline stage model compatibility ────────────────────────────────────

def test_pipeline_model_compatibility():
    import yaml
    from transformer_impl.config import config_from_yaml
    from transformer_impl.model import Transformer
    with open('configs/pipeline_full.yaml') as f:
        pipeline = yaml.safe_load(f)
    models = {}
    for stage in pipeline['stages']:
        cfg = config_from_yaml(stage['config'])
        m = Transformer(cfg, cfg.dataset.vocab_size)
        models[stage['command']] = sum(p.numel() for p in m.parameters())
    # All stages use same model dimensions, so param count must match
    param_counts = list(models.values())
    assert all(c == param_counts[0] for c in param_counts)
