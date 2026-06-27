# Resultados

## Modelo

| Atributo | Valor |
|----------|-------|
| Parámetros | 13,326,464 (13.33M) |
| Arquitectura | MHA + SwiGLU + sin posición |
| Dimensiones | d_model=256, num_layers=6, num_heads=4, d_ff=768 |
| Vocab | 16K BPE |
| Entrenamiento | pretrain → SFT → DPO |

## Pipeline

```
Pretraining ──► SFT ──► DPO
(tinystories)  (alpaca)  (ultrafeedback)
```

## Evaluación por etapa (cada una en su tarea)

| Etapa | Evaluado en | Tarea | Params | Loss | PPL | Generación |
|-------|-------------|-------|--------|------|-----|------------|
| SFT (10K steps) | Alpaca | Instruction Following | 13.33M | 0.4503 | **1.57** | *(ver nota)* |
| DPO (10K steps) | Alpaca | Preferencias | 13.33M | 0.4527 | **1.57** | *(ver nota)* |

> **Nota sobre generación de texto:**  
> Este modelo tiene solo **13M parámetros**. Es capaz de **predecir el siguiente token con alta precisión**
> (PPL 1.57 en tokens de respuesta), pero es **demasiado pequeño para generar texto coherente
> por sí mismo**. En la práctica, el modelo genera secuencias vacías o palabras sueltas sin orden.
> Para generación de texto útil se necesitan **100M+ parámetros** y órdenes de magnitud más datos.
> El valor del pipeline no es la calidad de generación, sino **demostrar el flujo completo**
> (pretrain → SFT → DPO → evaluación) funcionando de principio a fin.

## Evaluación completa

```bash
# Evaluar cada etapa en su tarea
python eval_checkpoint.py -m runs/pretrain_*/checkpoints/best_model.pt
python eval_checkpoint.py -m runs/sft_*/checkpoints/best_model.pt \
  -c configs/sft_alpaca.yaml -d alpaca_cleaned
python eval_checkpoint.py -m runs/dpo_*/checkpoints/best_model.pt \
  -c configs/sft_alpaca.yaml -d alpaca_cleaned

# Ver curvas en TensorBoard
tensorboard --logdir runs/
```
