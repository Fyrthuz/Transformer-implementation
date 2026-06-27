# Training Pipeline

## Pipeline Completo

```
Pretraining ──► SFT ──► DPO/PPO ──► (opcional) GRPO
(from scratch)  (instruct)  (alignment)    (reasoning)
```

## Comandos CLI

| Comando | Descripción | Modelo base |
|---------|-------------|-------------|
| `pretrain` | Pre-training desde cero | Ninguno |
| `sft` | Supervised fine-tuning | Preentrenado |
| `dpo` | Direct Preference Optimization | SFT |
| `ppo` | PPO alignment | SFT |
| `grpo` | Group Relative Policy Optimization | SFT |
| `pipeline` | Multi-stage pipeline (encadena stages) | Ver stages |

## Fine-tuning con Configuración Diferente

El pipeline soporta cargar un checkpoint pre-entrenado aunque la **nueva configuración tenga arquitectura distinta** (d_model, num_layers, num_heads, vocab_size). Ver [Training docs](training.md#fine-tuning-con-checkpoint-adaptativo).

```bash
# Pre-entrenar con modelo 6 layers, fine-tunear con 8 layers
python run.py pretrain -c configs/pretrain_tinystories.yaml model.num_layers=6
python run.py sft -c configs/sft_alpaca.yaml model.num_layers=8 -m best_model.pt
```

## Pipeline Multi-etapa (Recomendado)

Un solo comando ejecuta las etapas secuencialmente, lanza TensorBoard automáticamente:

```bash
# 3 etapas (rápido, recomendado):
./run_experiment.sh pipeline -c configs/pipeline_sft_dpo.yaml
# 4 etapas (incluye GRPO, ~30s/step):
./run_experiment.sh pipeline -c configs/pipeline_full.yaml
# o directamente:
python run.py pipeline -c configs/pipeline_sft_dpo.yaml
```

### pipeline_sft_dpo.yaml

```yaml
name: sft_dpo_pipeline
stages:
  - name: "Pre-training (TinyStories)"
    command: pretrain
    config: configs/pretrain_tinystories.yaml
  - name: "SFT (Alpaca)"
    command: sft
    config: configs/sft_alpaca.yaml
  - name: "DPO (UltraFeedback)"
    command: dpo
    config: configs/dpo_ultrafeedback.yaml
```

### pipeline_full.yaml (con GRPO)

```yaml
name: full_pipeline
stages:
  - name: "Pre-training (TinyStories)"
    command: pretrain
    config: configs/pretrain_tinystories.yaml
  - name: "SFT (Alpaca)"
    command: sft
    config: configs/sft_alpaca.yaml
  - name: "DPO (UltraFeedback)"
    command: dpo
    config: configs/dpo_ultrafeedback.yaml
  - name: "GRPO (GSM8K Reasoning)"
    command: grpo
    config: configs/grpo_gsm8k.yaml
```

Cada etapa recibe automáticamente el `best_model.pt` de la etapa anterior como checkpoint inicial. Se pueden combinar stages en cualquier orden.

## Ejecución Manual (etapa por etapa)

```bash
# 1. Pre-training
python run.py pretrain -c configs/pretrain_tinystories.yaml

# 2. SFT (checkpoint de pretrain está en runs/.../checkpoints/)
python run.py sft -c configs/sft_alpaca.yaml -m runs/pretrain_*/checkpoints/best_model.pt

# 3. DPO
python run.py dpo -c configs/dpo_ultrafeedback.yaml -m runs/sft_*/checkpoints/best_model.pt
```

> El pipeline (`./run_experiment.sh pipeline -c configs/pipeline_sft_dpo.yaml`) pasa automáticamente el `best_model.pt` de cada etapa a la siguiente sin necesidad de rutas manuales.

## Carga Adaptativa de Checkpoints

Al usar `-m checkpoint.pt`, el sistema adapta automáticamente el checkpoint a la configuración actual:

- **Mismo `vocab_size`** → carga completa
- **`vocab_size` diferente** → redimensiona embedding y output layer, preservando pesos compartidos
- **`num_layers` diferente** → carga capas comunes (primeras N), resto aleatorio
- **`d_model` o `num_heads` diferente** → omite capas con shape mismatch

Esto permite fine-tuning entre arquitecturas distintas sin reentrenar desde cero.

## TensorBoard Logging

Cada etapa loggea automáticamente en `runs/` con nombre descriptivo:

```
{etapa}_{attention}_{ffn}_{position}_d{d_model}l{num_layers}_{timestamp}
```

Ejemplo: `pipeline_sft_mha_swiglu_none_d256l6_20260626_202617/`

Override con `name=custom_name` en CLI overrides.

| Métrica | Frecuencia | Descripción |
|---------|-----------|-------------|
| `Train/Loss` | cada log_interval steps | Pérdida de entrenamiento |
| `Train/LR` | cada log_interval steps | Learning rate actual |
| `Train/GradNorm` | cada log_interval steps | Norma del gradiente |
| `Train/StepTime_ms` | cada log_interval steps | Tiempo por step |
| `Eval/Loss` | cada eval_steps | Pérdida en validación |
| `Eval/Perplexity` | cada eval_steps | Perplejidad |
| `Samples/Generated` | cada eval_steps | Texto generado como ejemplo |
| `grad/*` | cada 10×log_interval steps | Histogramas de gradientes |
| `param/*` | cada 10×log_interval steps | Histogramas de pesos |
| `GPU/Memory_*` | cada log_interval steps | Memoria GPU |

Para ver en vivo:

```bash
tensorboard --logdir runs
```

## Arquitectura de Trainers

```
BaseTrainer (compartido)
  ├── PreTrainer (pretrain)
  ├── SFTTrainer (sft)
  ├── DPOTrainer (dpo)
  ├── PPOTrainer (ppo)
  └── GRPOTrainer (grpo)
```

Todos heredan de `BaseTrainer` que proporciona:
- Optimizador y LR scheduler (warmup_cosine, warmup_constant, linear, cosine)
- AMP (mixed precision fp16/bf16)
- Gradient accumulation
- Checkpointing (automático cada save_steps)
- Logging (TensorBoard + WandB)
- Evaluación periódica con generación de samples
- Histogramas de gradientes y pesos
