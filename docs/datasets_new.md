# Catálogo de Datasets

Todos los datasets se descargan automáticamente al ejecutar el experimento.

## Pre-training

| Nombre | ID | # Ejemplos | Licencia |
|--------|----|-----------|----------|
| TinyStories | `tinystories` | 2.1M cuentos | CDLA-Sharing |
| FineWeb | `fineweb` | ~10B tokens sample | ODC-BY |
| WikiText-2 | `wikitext2` | ~2M tokens | Creative Commons |
| TinyShakespeare | `tinyshakespeare` | ~1M chars | Público |

## SFT (Supervised Fine-Tuning)

| Nombre | ID | # Ejemplos | Licencia |
|--------|----|-----------|----------|
| Alpaca Cleaned | `alpaca_cleaned` | 52K | CC BY NC 4.0 |
| OpenAssistant | `oasst1` | 88K conversaciones | Apache 2.0 |
| Dolly | `dolly` | 15K | CC BY-SA 3.0 |

## Preferencia (DPO/PPO)

| Nombre | ID | # Pares | Licencia |
|--------|----|---------|----------|
| UltraFeedback Binarized | `ultrafeedback` | 61K | MIT |
| Anthropic HH-RLHF | `hh_rlhf` | 170K | MIT |

## Reasoning (GRPO)

| Nombre | ID | # Problemas | Tipo | Licencia |
|--------|----|------------|------|----------|
| GSM8K | `gsm8k` | 8.5K | Matemáticas | MIT |
| MATH | `math` | 12.5K | Matemáticas avanzadas | MIT |
| MBPP | `mbpp` | 974 | Programación | CC BY 4.0 |

## Uso

En el YAML:
```yaml
dataset:
  name: alpaca_cleaned
  tokenization: bpe
  vocab_size: 16000
  max_seq_len: 512
```
