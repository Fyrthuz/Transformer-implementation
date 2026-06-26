import os
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir=None, name="default", log_type="tensorboard"):
        self.log_type = log_type
        self.writer = None
        self.wandb_run = None
        if log_type == "tensorboard":
            self.writer = SummaryWriter(log_dir=log_dir or f'runs/{name}')
        elif log_type == "wandb":
            try:
                import wandb
                self.wandb_run = wandb.init(project=name, dir=log_dir)
            except ImportError:
                print("wandb not installed, falling back to tensorboard")
                self.writer = SummaryWriter(log_dir=log_dir or f'runs/{name}')
                self.log_type = "tensorboard"

    def log_scalar(self, tag, value, step):
        if self.writer:
            self.writer.add_scalar(tag, value, step)
        elif self.wandb_run:
            self.wandb_run.log({tag: value}, step=step)

    def log_text(self, tag, text, step):
        if self.writer:
            self.writer.add_text(tag, text, step)

    def log_hparams(self, hparam_dict, metric_dict):
        if self.writer:
            self.writer.add_hparams(hparam_dict, metric_dict)

    def log_histogram(self, tag, values, step):
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_figure(self, tag, figure, step):
        if self.writer:
            self.writer.add_figure(tag, figure, step)

    def log_scalars(self, main_tag, tag_value_dict, step):
        if self.writer:
            self.writer.add_scalars(main_tag, tag_value_dict, step)

    def log_gradient_histograms(self, model, step):
        if not self.writer:
            return
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'grad/{name}', param.grad, step)
            if param.requires_grad:
                self.writer.add_histogram(f'param/{name}', param.data, step)

    def log_model_graph(self, model, input_example):
        if self.writer:
            try:
                self.writer.add_graph(model, input_example)
            except Exception:
                pass

    def flush(self):
        if self.writer:
            self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()
        if self.wandb_run:
            self.wandb_run.finish()
