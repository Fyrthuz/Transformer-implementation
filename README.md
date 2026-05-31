# Transformer Experiment: Tiny Shakespeare

Este proyecto implementa un modelo **Transformer Decoder-only** (estilo GPT) entrenado para tareas de modelado de lenguaje causal (generación de texto) utilizando el dataset de Shakespeare.

## Estructura del Proyecto

- **`main.py`**: El script principal que coordina la carga de datos, el entrenamiento y la validación. Incluye integración con TensorBoard.
- **`transformer.py`**: Definición de la arquitectura. Incluye Multi-Head Attention, bloques de Transformer, codificación posicional y capas de embedding.
- **`dataset_preparation.py`**: Script para descargar y preparar el dataset `tinyshakespeare.txt`. Contiene la lógica de `SimpleTokenizer` y el split de datos.
- **`generate.py`**: Script para realizar inferencia y generar texto nuevo a partir de un prompt usando el mejor checkpoint guardado.
- **`requirements.txt`**: Dependencias del proyecto.

## Requisitos

Instala las dependencias necesarias con:
```bash
pip install -r requirements.txt
```

## Entrenamiento

Para iniciar el entrenamiento, simplemente ejecuta:
```bash
python main.py
```
El modelo guardará los logs de entrenamiento en la carpeta `runs/`.

## Monitoreo con TensorBoard

Para visualizar las métricas (Loss de entrenamiento y test) en tiempo real:
1. Abre una terminal en la raíz del proyecto.
2. Ejecuta:
   ```bash
   tensorboard --logdir=runs
   ```
3. Abre `http://localhost:6006` en tu navegador.

## Generación de Texto

Para probar el modelo entrenado y generar diálogos al estilo Shakespeare:
```bash
python generate.py
```
Puedes ajustar la `temperature` en el script para controlar la creatividad de la salida.

## Detalles Técnicos
- **Arquitectura**: 4 capas, 8 cabezales de atención, 256 dimensiones de embedding, 1024 d_ff.
- **Contexto**: Longitud de secuencia máxima de 128 caracteres.
- **Dataset**: Tiny Shakespeare (90% entrenamiento / 10% validación). Se utiliza una ventana deslizante con stride de 16 para aumentar el volumen de datos de entrenamiento.
- **Tokenización**: Nivel de carácter (Vocabulario reducido de ~66 tokens), lo que permite al modelo aprender ortografía y estructura desde cero.