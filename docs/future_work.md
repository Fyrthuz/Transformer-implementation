# Future Work — No Implementado

Las siguientes funcionalidades están fuera del alcance actual pero son candidatas para futuras iteraciones:

| Funcionalidad | Razón |
|--------------|-------|
| **KV-cache para inferencia** | No crítico para 13M params; generación suficientemente rápida sin él |
| **FlashAttention** | Requiere GPUs Ampere+; añade complejidad innecesaria para 13M params |
| **FSDP / Distributed Data Parallel** | 13M params cabe en 1 GPU; se puede añadir vía `torchrun` |
| **Tensor Parallelism / Pipeline Parallelism** | Solo necesario para modelos >7B |
| **LoRA / QLoRA (PEFT)** | El repo entrena full fine-tuning; LoRA sería otra etapa |
| **Gradient Checkpointing** | Añade overhead computacional; no necesario para 13M params |
| **Serving / API REST** | Fuera del alcance educativo/experimental |
| **Evaluación benchmarks (MMLU, HellaSwag, etc.)** | Requiere librería `lm-eval-harness` |
| **Continuous batching** | Solo relevante para producción |
| **Quantization (GPTQ, AWQ)** | Para despliegue; fuera de alcance |
| **Verificador externo para GRPO** | Actualmente usa exact_match; un verificador más sofisticado mejoraría resultados |
| **Reward model entrenado para PPO** | Actualmente usa reward simulada; un RM real mejoraría la alineación |
